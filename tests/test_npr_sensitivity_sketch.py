"""NPR coarse KL, VJP sensitivity, logit-contrast sketch folding."""

from __future__ import annotations

import pytest
import torch

from aimo.native_extract import (
    AuditSink,
    PromptPolicy,
    make_stage_forward,
    render_prompt,
    run_extraction,
    tokenize_prompt,
)
from aimo.native_toy import ToyTokenizer, build_toy_hybrid
from aimo.npr import (
    chunked_logsumexp,
    coarse_kl,
    full_vocab_kl,
    problem_normalized_mean,
    student_coarse,
    teacher_topk_other,
)
from aimo.sensitivity import (
    directional_fd_check,
    probe_vectors,
    response_vjp,
    sensitivity_loss,
)
from aimo.sketch import (
    FoldingInvalid,
    column_sum_error,
    fold_head,
    fold_student,
    make_sketch,
    verify_folding,
)


def _teacher(seed=0, n=6, h=5, v=50):
    g = torch.Generator().manual_seed(seed)
    hidden, weight = torch.randn(n, h, generator=g), torch.randn(v, h, generator=g)
    return hidden, weight, torch.randn(v, generator=g)


def test_teacher_topk_other_matches_dense_and_is_chunk_invariant():
    x, w, b = _teacher()
    dense = torch.log_softmax(x @ w.T + b, dim=-1)
    for chunk in (7, 16, 50):
        t = teacher_topk_other(x, w, b, k=8, chunk=chunk)
        ref_val, ref_idx = dense.topk(8, dim=-1)
        assert torch.allclose(t.logp, ref_val, atol=1e-5)
        assert torch.equal(t.indices, ref_idx)
        # OTHER는 나머지 vocabulary의 실제 질량입니다 (top-k를 재정규화하지 않습니다).
        other = 1 - ref_val.exp().sum(-1)
        assert torch.allclose(t.other_mass(), other, atol=1e-5)
        assert torch.allclose(t.logp.exp().sum(-1) + t.other_mass(), torch.ones(6), atol=1e-5)


def test_coarse_kl_zero_for_teacher_and_positive_otherwise():
    x, w, b = _teacher()
    teacher = teacher_topk_other(x, w, b, k=8, chunk=16)
    same = coarse_kl(teacher, *student_coarse(x, w, b, teacher, chunk=16))
    assert same.abs().max() < 1e-5
    other = coarse_kl(teacher, *student_coarse(x, w * 0.5, b, teacher, chunk=16))
    assert bool((other > 0).all())
    # coarse KL은 full KL의 하한입니다 (data processing inequality).
    full = full_vocab_kl(x, w, b, x, w * 0.5, b, chunk=16)
    assert bool((other <= full + 1e-5).all())


def test_full_vocab_kl_chunked_matches_dense():
    x, w, b = _teacher()
    lp = torch.log_softmax(x @ w.T + b, dim=-1)
    lq = torch.log_softmax(x @ (w * 0.7).T, dim=-1)
    dense = (lp.exp() * (lp - lq)).sum(-1)
    assert torch.allclose(full_vocab_kl(x, w, b, x, w * 0.7, None, chunk=9), dense, atol=1e-5)


def test_chunked_logsumexp_gradient_matches_dense():
    z = torch.randn(4, 3, requires_grad=True)
    d = torch.randn(40, 3, requires_grad=True)
    bias = torch.randn(40)
    chunked_logsumexp(z, d, bias, chunk=7).sum().backward()
    g_chunk = (z.grad.clone(), d.grad.clone())
    z.grad = d.grad = None
    torch.logsumexp(z @ d.T + bias, dim=-1).sum().backward()
    assert torch.allclose(g_chunk[0], z.grad, atol=1e-5)
    assert torch.allclose(g_chunk[1], d.grad, atol=1e-5)


def test_problem_normalization_does_not_count_tokens_as_samples():
    values = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
    assert float(problem_normalized_mean(values, mask)) == pytest.approx(2.0)
    staged = torch.stack([values, values], dim=1)
    assert float(problem_normalized_mean(staged, mask)) == pytest.approx(2.0)


def _toy_stage():
    model = build_toy_hybrid(n_layers=4)
    tok = ToyTokenizer()
    enc = tokenize_prompt(tok, render_prompt(tok, "If 2x = 6, find x.", PromptPolicy()), 256)
    ids = enc["input_ids"][None]
    mask = torch.ones_like(ids)
    sink = AuditSink(2)
    with torch.no_grad():
        run_extraction(model, ids, mask, sink, 2)
    sketch = make_sketch(48, q=6, seed=3)
    folded = fold_head(model, sketch, 3)
    return model, ids, mask, sink.stacked()["state"], folded


def test_vjp_matches_explicit_directional_derivative_and_detaches():
    model, ids, mask, state, folded = _toy_stage()
    forward = make_stage_forward(model, ids, mask, 2, folded)  # macro 1 입구 = layer 2 입력
    probes = probe_vectors(3, 6, seed=0)
    vjps = response_vjp(forward, state[:, 1], probes, mask)
    assert vjps.shape == (3, 1, ids.shape[1], 16)
    assert not vjps.requires_grad
    direction = torch.randn_like(state[:, 1])
    check = directional_fd_check(forward, state[:, 1], probes[0], mask, direction, step=1e-2)
    assert check["relative_error"] < 1e-2


def test_sensitivity_loss_equals_explicit_projector_and_no_second_order():
    torch.manual_seed(0)
    vjps = [torch.randn(3, 7, 10), torch.randn(3, 7, 10)]
    raw = torch.randn(2, 10, 4, requires_grad=True)
    basis = torch.stack([torch.linalg.qr(r)[0] for r in raw])
    loss = sensitivity_loss(vjps, basis)
    num = sum((a - a @ basis[g] @ basis[g].T).pow(2).sum() for g, a in enumerate(vjps))
    den = sum(a.pow(2).sum() for a in vjps)
    assert torch.allclose(loss, num / den, atol=1e-5)
    loss.backward()
    assert raw.grad is not None and all(not a.requires_grad for a in vjps)
    full = [torch.randn(3, 7, 10)] * 2
    eye_basis = torch.eye(10).expand(2, -1, -1)
    assert float(sensitivity_loss(full, eye_basis)) == pytest.approx(0.0, abs=1e-6)


def test_probes_are_fixed_by_seed():
    assert torch.equal(probe_vectors(4, 8, 1), probe_vectors(4, 8, 1))
    assert not torch.equal(probe_vectors(4, 8, 1), probe_vectors(4, 8, 2))


def test_sketch_zero_column_sum_and_folding_identity():
    sketch = make_sketch(48, q=8, seed=5)
    assert column_sum_error(sketch) < 1e-5
    assert not torch.allclose(sketch, make_sketch(48, q=8, seed=6))  # 독립 R'
    model = build_toy_hybrid(n_layers=4, head_bias=True)
    folded = fold_head(model, sketch, 5)
    ids = torch.tensor([[3, 5, 7, 9, 11]])

    def logits_fn():
        with torch.no_grad():
            normed = model.model(ids).last_hidden_state[0]
            return model.lm_head(normed), normed

    assert verify_folding(logits_fn, folded, sketch)["relative_error"] < 1e-4
    d, b = torch.randn(48, 4), torch.randn(48)
    w_s, b_s = fold_student(d, b, sketch)
    z = torch.randn(3, 4)
    student_logits = z @ d.T + b
    ref = torch.log_softmax(student_logits, -1) @ sketch
    assert torch.allclose(z @ w_s.T + b_s, ref, atol=1e-4)


def test_softcapped_head_is_rejected():
    model = build_toy_hybrid(n_layers=4)
    model.config.final_logit_softcapping = 30.0
    with pytest.raises(FoldingInvalid):
        fold_head(model, make_sketch(48, q=4, seed=1), 1)
