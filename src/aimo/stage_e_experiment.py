"""Stage-E discovery 학습, 연구 대조군, 평가 pipeline (offline 연구 경로).

teacher는 frozen target model의 native next-token distribution이며 label을 쓰지 않습니다.

    L = L_NPR + lambda * L_sensitivity

- `L_NPR`: teacher top-k + OTHER coarse KL (`npr`), stage 평균, 문제별 정규화.
- `L_sensitivity`: 고정 seed probe의 VJP capture (`sensitivity`), lambda=0 baseline 지원.
- rank와 lambda는 discovery Dev에서만, 미리 정한 규칙으로 고릅니다 (`select_on_dev`).
  held-out 결과를 보고 rank나 feature를 다시 고르는 반복을 만들지 않습니다.

대조군: random-r / PCA-r + trained decoder, NPR-only (lambda=0), full-hidden readout,
output-compression reduced-rank, support/query local-fit oracle.

평가:
A. held-out native fidelity와 gradient capture (output compression 대조군과 비교)
B. frozen encoder의 새 문제·topic·difficulty·MP family generalization
C. 고정 macro boundary에서 natural donor effect의 projected intervention recovery와
   random / complement / norm / rank / site-matched control. stage 하나만 patch하고, 여러 stage
   clamp를 자연적 causal mechanism으로 단정하지 않습니다.

native fidelity, causal use(C), 공식 robustness 예측 성능은 따로 보고합니다.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import StageEConfig
from .macro_page import macro_boundaries
from .native_data import (
    ROLE_DISCOVERY,
    DiscoveryProblem,
    assert_label_free,
    assign_splits,
    generalization_split,
    make_contrast_probes,
)
from .native_extract import (
    AuditSink,
    PromptPolicy,
    locate_backbone,
    make_stage_forward,
    pad_right,
    render_prompt,
    run_extraction,
    tokenize_prompt,
)
from .native_page import SEG_PROBE
from .npr import (
    TeacherTopK,
    coarse_kl,
    full_vocab_kl,
    problem_normalized_mean,
    student_coarse,
    teacher_topk_other,
)
from .sensitivity import probe_vectors, response_vjp, sensitivity_loss
from .sketch import FoldedHead, fold_head, make_sketch, output_head
from .stage_e import (
    CONTROL_PCA,
    CONTROL_RANDOM,
    CONTROL_REDUCED_RANK,
    CONTROL_STAGE_E,
    StageE,
    StageECheckpointMeta,
    output_compression_basis,
    pca_basis,
    random_basis,
    save_stage_e,
)

CONTROL_NPR_ONLY = "npr_only_lambda0"
CONTROL_FULL_HIDDEN = "full_hidden_readout"
CONTROL_LOCAL_ORACLE = "local_fit_oracle_support_query"
STATUS_TOY = "CPU_TOY_ONLY / NOT_RESEARCH_EVIDENCE"
EXPERIMENT_SCHEMA = "aimo-stage-e-experiment-v1"


@dataclass
class TeacherBatch:
    """frozen teacher가 한 batch에 대해 만든 관측. label을 담지 않습니다."""

    problem_ids: list[str]
    root_ids: list[str]
    state_out: Tensor  # [B, G, T, H] macro 출구 raw state
    mask: Tensor  # [B, T] bool
    segments: Tensor  # [B, T]
    normed: Tensor  # [B, T, H] final norm 이후 hidden
    teacher: TeacherTopK  # valid token을 row-major로 펼친 순서
    y: Tensor  # [B, T, q] native logit-contrast sketch
    y_audit: Tensor  # [B, T, q] 독립 sketch R'
    vjps: list[list[Tensor]] | None  # [B][G] -> [J, T_b, H]
    input_ids: list[Tensor]


class NativeTeacher:
    """frozen target model 하나에서 teacher batch를 만듭니다. 결과는 결정적이므로 cache합니다."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        cfg: StageEConfig,
        policy: PromptPolicy,
        *,
        topk: int | None = None,
        device: str = "cpu",
        cache: bool = True,
    ) -> None:
        self.model, self.tokenizer, self.cfg, self.policy = model, tokenizer, cfg, policy
        self.device = device
        self.use_cache = cache
        self.topk = topk or cfg.npr_topk
        self.backbone = locate_backbone(model)
        self.bounds = macro_boundaries(len(self.backbone.layers), cfg.n_macro)
        head = output_head(model)
        self.head_weight, self.head_bias = head.weight, head.bias
        vocab = head.weight.shape[0]
        self.sketch = make_sketch(vocab, cfg.sketch_q, cfg.sketch_seed)
        self.audit_sketch = make_sketch(vocab, cfg.sketch_q, cfg.audit_sketch_seed)
        self.folded: FoldedHead = fold_head(model, self.sketch, cfg.sketch_seed)
        self.folded_audit: FoldedHead = fold_head(model, self.audit_sketch, cfg.audit_sketch_seed)
        self.probes = probe_vectors(cfg.n_probes, cfg.sketch_q, cfg.probe_seed)
        self._cache: dict[tuple, TeacherBatch] = {}

    def batch(self, records: list[dict], need_vjp: bool) -> TeacherBatch:
        key = (tuple(r["problem_id"] for r in records),
               tuple(r.get("probe_text") or "" for r in records), need_vjp)
        if not self.use_cache:
            return self._build(records, need_vjp)
        if key not in self._cache:
            self._cache[key] = self._build(records, need_vjp)
        return self._cache[key]

    def _build(self, records: list[dict], need_vjp: bool) -> TeacherBatch:
        encoded = []
        for record in records:
            assert_label_free(record)
            rendered = render_prompt(self.tokenizer, record["text"], self.policy,
                                     probe_text=record.get("probe_text"))
            encoded.append(tokenize_prompt(self.tokenizer, rendered, self.policy.max_tokens))
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        ids, mask = pad_right([e["input_ids"] for e in encoded], pad_id)
        segments = torch.full(ids.shape, -1, dtype=torch.long)
        for i, e in enumerate(encoded):
            segments[i, : e["segments"].shape[0]] = e["segments"]
        sink = AuditSink(self.cfg.n_macro, device=self.device, states_only=True)
        with torch.no_grad():
            run_extraction(self.model, ids.to(self.device), mask.to(self.device), sink,
                           self.cfg.n_macro)
        state = sink.stacked()["state"]
        normed = sink.final_norm
        maskb = mask.bool().to(self.device)
        teacher = teacher_topk_other(normed[maskb], self.head_weight, self.head_bias, self.topk,
                                     self.cfg.vocab_chunk)
        vjps = None
        if need_vjp:
            vjps = []
            for b, e in enumerate(encoded):
                t = int(e["input_ids"].shape[0])
                ids_b = e["input_ids"][None].to(self.device)
                mask_b = torch.ones_like(ids_b)
                per_stage = []
                for g in range(self.cfg.n_macro):
                    forward = make_stage_forward(self.model, ids_b, mask_b, self.bounds[g + 1],
                                                 self.folded)
                    grads = response_vjp(forward, state[b, g + 1, :t][None], self.probes, mask_b)
                    per_stage.append(grads[:, 0])
                vjps.append(per_stage)
        with torch.no_grad():
            y, y_audit = self.folded(normed), self.folded_audit(normed)
        return TeacherBatch(
            problem_ids=[r["problem_id"] for r in records],
            root_ids=[r["root_id"] for r in records],
            state_out=state[:, 1:],
            mask=maskb,
            segments=segments.to(self.device),
            normed=normed,
            teacher=teacher,
            y=y,
            y_audit=y_audit,
            vjps=vjps,
            input_ids=[e["input_ids"] for e in encoded],
        )


def _relative_sq(estimate: Tensor, reference: Tensor) -> Tensor:
    """token별 ||estimate - reference||² / ||reference||²."""
    return (estimate - reference).pow(2).sum(-1) / reference.pow(2).sum(-1).clamp_min(1e-12)


def _scatter(values: Tensor, mask: Tensor) -> Tensor:
    out = torch.zeros(mask.shape, device=values.device, dtype=values.dtype)
    out[mask] = values
    return out


def losses(model: StageE, batch: TeacherBatch, lam: float, cfg: StageEConfig) -> dict[str, Tensor]:
    z = model.encode_state(batch.state_out)  # [B, G, T, r]
    per_stage = []
    for g in range(model.n_macro):
        logp_k, log_other = student_coarse(z[:, g][batch.mask], model.decoder,
                                           model.decoder_bias, batch.teacher, cfg.vocab_chunk)
        per_stage.append(_scatter(coarse_kl(batch.teacher, logp_k, log_other), batch.mask))
    npr = problem_normalized_mean(torch.stack(per_stage, dim=1), batch.mask)
    out = {"npr": npr, "total": npr}
    if batch.vjps is not None:
        basis = model.basis()
        sens = torch.stack([sensitivity_loss(v, basis, cfg.sensitivity_eps) for v in batch.vjps])
        out["sensitivity"] = sens.mean()
        if lam > 0:
            out["total"] = npr + lam * out["sensitivity"]
    elif lam > 0:
        raise ValueError("positive lambda needs VJP teacher batches")
    return out


def train_state_sample(teacher: NativeTeacher, train_chunks: list[list[dict]],
                       max_tokens: int = 20000) -> tuple[Tensor, Tensor]:
    """PCA 대조군용 train state 표본 ([N, G, 1, H], mask [N, 1]). train split만 씁니다."""
    rows = []
    for chunk in train_chunks:
        batch = teacher.batch(chunk, need_vjp=False)
        rows.append(batch.state_out.permute(0, 2, 1, 3)[batch.mask].cpu())  # [n, G, H]
        if sum(r.shape[0] for r in rows) >= max_tokens:
            break
    flat = torch.cat(rows)
    step = max(1, flat.shape[0] // max_tokens)
    flat = flat[::step][:max_tokens]
    return flat.unsqueeze(2), torch.ones(flat.shape[0], 1, dtype=torch.bool)


def make_model(control: str, cfg: StageEConfig, hidden: int, vocab: int, rank: int, seed: int,
               train_sample: tuple[Tensor, Tensor], teacher: NativeTeacher) -> StageE:
    """대조군별 encoder 초기화. fixed-basis 대조군은 train split 통계만 씁니다."""
    g = cfg.n_macro
    if control in (CONTROL_STAGE_E, CONTROL_NPR_ONLY):
        return StageE(g, hidden, rank, vocab, seed=seed)
    if control == CONTROL_RANDOM:
        return StageE(g, hidden, rank, vocab, seed=seed,
                      init_basis=random_basis(g, hidden, rank, seed), freeze_basis=True)
    if control == CONTROL_PCA:
        basis = pca_basis(train_sample[0], train_sample[1], rank)
        return StageE(g, hidden, rank, vocab, seed=seed, init_basis=basis, freeze_basis=True)
    if control == CONTROL_REDUCED_RANK:
        basis = output_compression_basis(teacher.folded.weight, g, rank)
        return StageE(g, hidden, rank, vocab, seed=seed, init_basis=basis, freeze_basis=True)
    if control == CONTROL_FULL_HIDDEN:
        eye = torch.eye(hidden).expand(g, -1, -1).clone()
        return StageE(g, hidden, hidden, vocab, seed=seed, init_basis=eye, freeze_basis=True)
    raise ValueError(f"unknown control {control!r}")


def train_stage_e(
    model: StageE,
    teacher: NativeTeacher,
    train_chunks: list[list[dict]],
    dev_chunks: list[list[dict]],
    lam: float,
    cfg: StageEConfig,
    *,
    epochs: int,
    lr: float = 2e-2,
    seed: int = 0,
) -> dict:
    """chunk 단위 Adam. checkpoint는 discovery Dev의 같은 objective로만 고릅니다."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    rng = random.Random(f"stage-e-train:{seed}")
    need_vjp = lam > 0
    best = (float("inf"), None, -1)
    history = []
    for epoch in range(epochs):
        order = list(range(len(train_chunks)))
        rng.shuffle(order)
        train_total = 0.0
        for index in order:
            batch = teacher.batch(train_chunks[index], need_vjp=need_vjp)
            optimizer.zero_grad()
            loss = losses(model, batch, lam, cfg)
            loss["total"].backward()
            optimizer.step()
            train_total += float(loss["total"]) / len(order)
        with torch.no_grad():
            weights, values = [], []
            for chunk in dev_chunks:
                dev_batch = teacher.batch(chunk, need_vjp)
                values.append(float(losses(model, dev_batch, lam, cfg)["total"]))
                weights.append(len(chunk))
        value = sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights)
        history.append({"epoch": epoch, "train_total": train_total, "dev_total": value})
        if value < best[0]:
            best = (value, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    return {"best_epoch": best[2], "best_dev_total": best[0], "history_tail": history[-3:]}


def evaluate(model: StageE, teacher: NativeTeacher, chunks: list[list[dict]], cfg: StageEConfig,
             need_vjp: bool = True) -> dict:
    """chunk별 fidelity를 문제 수로 가중 평균합니다 (문제별 정규화를 유지합니다)."""
    parts = [(len(chunk), fidelity(model, teacher.batch(chunk, need_vjp), teacher, cfg))
             for chunk in chunks]
    total = sum(n for n, _ in parts)
    out = {}
    for key, value in parts[0][1].items():
        if isinstance(value, list):
            out[key] = [sum(n * m[key][i] for n, m in parts) / total for i in range(len(value))]
        else:
            out[key] = sum(n * m[key] for n, m in parts) / total
    out["n_problems"] = total
    return out


@torch.no_grad()
def fidelity(model: StageE, batch: TeacherBatch, teacher: NativeTeacher, cfg: StageEConfig) -> dict:
    """A. native fidelity와 gradient capture. 모든 값은 문제별 정규화 뒤 평균입니다."""
    z = model.encode_state(batch.state_out)
    coarse, full, sketch_err, audit_err = [], [], [], []
    w_s, b_s = model.folded_decoder(teacher.sketch.to(z.device))
    w_a, b_a = model.folded_decoder(teacher.audit_sketch.to(z.device))
    normed_flat = batch.normed[batch.mask]
    for g in range(model.n_macro):
        zg = z[:, g][batch.mask]
        logp_k, log_other = student_coarse(zg, model.decoder, model.decoder_bias, batch.teacher,
                                           cfg.vocab_chunk)
        coarse.append(_scatter(coarse_kl(batch.teacher, logp_k, log_other), batch.mask))
        full.append(_scatter(full_vocab_kl(normed_flat, teacher.head_weight, teacher.head_bias,
                                           zg, model.decoder, model.decoder_bias,
                                           cfg.vocab_chunk), batch.mask))
        y_hat = z[:, g] @ w_s.T + b_s
        y_hat_a = z[:, g] @ w_a.T + b_a
        sketch_err.append(_relative_sq(y_hat, batch.y))
        audit_err.append(_relative_sq(y_hat_a, batch.y_audit))
    stack = lambda xs: torch.stack(xs, dim=1)  # noqa: E731
    out = {
        "coarse_topk_other_kl": float(problem_normalized_mean(stack(coarse), batch.mask)),
        "full_vocab_kl_audit": float(problem_normalized_mean(stack(full), batch.mask)),
        "sketch_relative_error": float(problem_normalized_mean(stack(sketch_err), batch.mask)),
        "independent_sketch_relative_error": float(
            problem_normalized_mean(stack(audit_err), batch.mask)),
        "coarse_kl_by_stage": [float(problem_normalized_mean(c.unsqueeze(1), batch.mask))
                               for c in coarse],
    }
    if batch.vjps is not None:
        basis = model.basis()
        sens = torch.stack([sensitivity_loss(v, basis, cfg.sensitivity_eps) for v in batch.vjps])
        out["gradient_capture"] = float(1.0 - sens.mean())
    return out


@torch.no_grad()
def local_fit_oracle(batch: TeacherBatch, model: StageE, teacher: NativeTeacher, rank: int) -> dict:
    """support(짝수 token)로 문제별 basis와 sketch decoder를 맞추고 query(홀수 token)에서 봅니다.

    test-time fitting을 쓰는 oracle 참고값이며 제출 경로에서는 절대 쓰지 않습니다.
    """
    oracle_err, oracle_cap, frozen_err, frozen_cap = [], [], [], []
    w_s, b_s = model.folded_decoder(teacher.sketch.to(batch.y.device))
    frozen_basis = model.basis()
    z_all = model.encode_state(batch.state_out)
    for b in range(batch.state_out.shape[0]):
        valid = batch.mask[b].nonzero().flatten()
        support, query = valid[0::2], valid[1::2]
        if support.numel() <= rank or query.numel() == 0:
            continue
        for g in range(batch.state_out.shape[1]):
            s = batch.state_out[b, g]
            mu = s[support].mean(0, keepdim=True)
            _, _, vh = torch.linalg.svd(s[support] - mu, full_matrices=False)
            basis = vh[:rank].T
            feats = torch.cat([(s - mu) @ basis, torch.ones(s.shape[0], 1)], dim=1)
            sol = torch.linalg.lstsq(feats[support], batch.y[b, support]).solution
            pred = feats[query] @ sol
            y_q = batch.y[b, query]
            oracle_err.append(float((pred - y_q).pow(2).sum() / y_q.pow(2).sum().clamp_min(1e-12)))
            y_hat = z_all[b, g, query] @ w_s.T + b_s
            frozen_err.append(float((y_hat - y_q).pow(2).sum() / y_q.pow(2).sum().clamp_min(1e-12)))
            if batch.vjps is not None:
                a = batch.vjps[b][g][:, query]
                total = a.pow(2).sum().clamp_min(1e-12)
                oracle_cap.append(float((a @ basis).pow(2).sum() / total))
                frozen_cap.append(float((a @ frozen_basis[g]).pow(2).sum() / total))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "oracle_query_sketch_relative_error": mean(oracle_err),
        "frozen_query_sketch_relative_error": mean(frozen_err),
        "oracle_query_gradient_capture": mean(oracle_cap),
        "frozen_query_gradient_capture": mean(frozen_cap),
        "note": "oracle refits per problem on support tokens; never used for submission",
    }


def select_on_dev(results: list[dict], tolerance: float) -> dict:
    """미리 정한 선택 규칙 (discovery Dev 결과만 받습니다).

    1. rank: dev coarse KL이 최선값의 (1 + tolerance) 안에 드는 가장 작은 rank.
    2. lambda: 그 rank에서 dev coarse KL이 lambda=0 대비 (1 + tolerance) 안에 드는 후보 중
       dev gradient capture가 가장 큰 값. 동률이면 작은 lambda.
    """
    if any(r.get("split") != "dev" for r in results):
        raise ValueError("selection must only see discovery Dev results")
    stage = [r for r in results if r["control"] == CONTROL_STAGE_E]
    best_kl = min(r["metrics"]["coarse_topk_other_kl"] for r in stage)
    ranks = sorted({r["rank"] for r in stage
                    if r["metrics"]["coarse_topk_other_kl"] <= best_kl * (1 + tolerance)})
    rank = ranks[0]
    at_rank = [r for r in stage if r["rank"] == rank]
    base = [r for r in at_rank if r["lambda"] == 0.0]
    base_kl = base[0]["metrics"]["coarse_topk_other_kl"] if base else best_kl
    ok = [r for r in at_rank if r["metrics"]["coarse_topk_other_kl"] <= base_kl * (1 + tolerance)]
    chosen = max(ok, key=lambda r: (r["metrics"].get("gradient_capture", 0.0), -r["lambda"]))
    return {"rank": rank, "lambda": chosen["lambda"], "rule": select_on_dev.__doc__.strip(),
            "tolerance": tolerance}


def _patched_response(teacher: NativeTeacher, ids: Tensor, boundary: int, site: int,
                      value: Tensor) -> Tensor:
    """boundary layer 입력의 한 site를 value로 바꾸고 native sketch [T, q]를 돌려줍니다."""
    backbone = teacher.backbone
    target = (backbone.final_norm if boundary == len(backbone.layers)
              else backbone.layers[boundary])
    captured = {}

    def replace(_module, args, kwargs):  # noqa: ANN001
        hidden = kwargs.get("hidden_states") if kwargs and "hidden_states" in kwargs else args[0]
        hidden = hidden.clone()
        hidden[0, site] = value.to(hidden.dtype)
        if kwargs and "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hidden
            return args, kwargs
        return (hidden, *args[1:]), kwargs

    def grab(_module, _args, output):  # noqa: ANN001
        captured["normed"] = output[0] if isinstance(output, tuple) else output

    handles = [target.register_forward_pre_hook(replace, with_kwargs=True),
               backbone.final_norm.register_forward_hook(grab)]
    try:
        with torch.no_grad():
            backbone.module(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return teacher.folded(captured["normed"][0].float())


@torch.no_grad()
def intervention_recovery(
    model: StageE, batch: TeacherBatch, teacher: NativeTeacher, stage: int, seed: int,
    pca: Tensor,
) -> dict:
    """C. natural donor effect의 projected intervention recovery.

    recipient i의 prompt-end site에 donor j(다른 root)의 같은 site state를 넣은 full patch 효과를
    기준으로, `E^+ (z_donor - z_recipient) = B B^T Δs` patch가 그 효과를 얼마나 회복하는지
    봅니다. control: random subspace, complement, norm-matched random, rank-matched PCA,
    site-matched random donor(같은 site에 donor의 다른 token state).
    """
    rng = random.Random(f"intervention:{seed}:{stage}")
    generator = torch.Generator().manual_seed(seed)
    boundary = teacher.bounds[stage + 1]
    basis = model.basis()[stage]
    rand = random_basis(1, basis.shape[0], basis.shape[1], seed)[0]
    names = ("projected", "random_subspace", "complement", "norm_matched_random",
             "rank_matched_pca", "site_matched_random_donor")
    scores: dict[str, list[float]] = {name: [] for name in names}
    n = batch.state_out.shape[0]
    for i in range(n):
        donors = [j for j in range(n) if batch.root_ids[j] != batch.root_ids[i]]
        if not donors:
            continue
        j = rng.choice(donors)
        site_i = int(batch.mask[i].sum()) - 1
        site_j = int(batch.mask[j].sum()) - 1
        ids = batch.input_ids[i][None].to(batch.state_out.device)
        s_r = batch.state_out[i, stage, site_i]
        delta = batch.state_out[j, stage, site_j] - s_r
        base = teacher.folded(batch.normed[i, : site_i + 1])[site_i]
        full = _patched_response(teacher, ids, boundary, site_i, s_r + delta)[site_i] - base
        denom = full.pow(2).sum().clamp_min(1e-12)
        proj = (delta @ basis) @ basis.T
        other_site = rng.choice([t for t in range(int(batch.mask[j].sum())) if t != site_j]
                                or [site_j])
        delta_other = batch.state_out[j, stage, other_site] - s_r
        direction = torch.randn(delta.shape, generator=generator)
        candidates = {
            "projected": proj,
            "random_subspace": (delta @ rand) @ rand.T,
            "complement": delta - proj,
            "norm_matched_random": direction / direction.norm() * proj.norm(),
            "rank_matched_pca": (delta @ pca[stage]) @ pca[stage].T,
            "site_matched_random_donor": (delta_other @ basis) @ basis.T,
        }
        for name, patch in candidates.items():
            effect = _patched_response(teacher, ids, boundary, site_i, s_r + patch)[site_i] - base
            scores[name].append(float(1.0 - (effect - full).pow(2).sum() / denom))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "stage": stage,
        "boundary_layer": boundary,
        "n_pairs": len(scores["projected"]),
        "recovery": {name: mean(values) for name, values in scores.items()},
        "note": ("single-stage patch at a frozen macro boundary; not a claim that multi-stage "
                 "clamps are the model's natural causal mechanism"),
    }


def toy_problems(n_roots: int = 24, seed: int = 0) -> list[DiscoveryProblem]:
    """가변 길이의 합성 수학 문장. 공백과 줄바꿈을 섞어 MP view가 실제로 달라지게 합니다."""
    rng = random.Random(f"toy-problems:{seed}")
    topics = ("algebra", "number_theory", "combinatorics")
    problems = []
    for i in range(n_roots):
        a, b, c = rng.randint(2, 9), rng.randint(1, 20), rng.randint(21, 60)
        pieces = [f"Let x  be a real number.\nIf {a}x + {b} = {c},", "find  x."]
        if rng.random() < 0.5:
            pieces.insert(1, f"and x > -{b}\nholds,")
        if rng.random() < 0.3:
            pieces.append("Give  the exact value.")
        problems.append(DiscoveryProblem(
            problem_id=f"toy:{i}", root_id=f"toy-root:{i}", source="aimo/toy", role=ROLE_DISCOVERY,
            text=" ".join(pieces),
            meta={"topic": topics[i % 3], "difficulty": 1 + (i // 3) % 3},
        ))
    return problems


def _records(problems: list[DiscoveryProblem]) -> list[dict]:
    return [p.record() for p in problems]


def chunked(records: list[dict], size: int) -> list[list[dict]]:
    return [records[i : i + size] for i in range(0, len(records), size)]


def run_discovery(
    teacher: NativeTeacher,
    by_split: dict[str, list[DiscoveryProblem]],
    cfg: StageEConfig,
    run_dir: Path,
    *,
    epochs: int,
    seed: int,
    batch_size: int,
    native_meta: dict,
    dataset_meta: dict,
    status: str,
    intervention_limit: int = 16,
) -> dict:
    """grid 학습(Dev 선택) -> freeze -> 대조군 -> A/B/C 평가. toy와 server가 같은 경로를 씁니다."""
    started = time.monotonic()
    train_chunks = chunked(_records(by_split["train"]), batch_size)
    dev_chunks = chunked(_records(by_split["dev"]), batch_size)
    held_chunks = chunked(_records(by_split["held_out"]), batch_size)
    first = teacher.batch(train_chunks[0], need_vjp=False)
    hidden, vocab = first.state_out.shape[-1], teacher.head_weight.shape[0]
    sample = train_state_sample(teacher, train_chunks)
    ranks = [r for r in cfg.rank_grid if r < hidden]
    if not ranks:
        raise ValueError(f"no rank in {cfg.rank_grid} is smaller than H={hidden}")

    dev_results = []
    for rank in ranks:
        for lam in cfg.lambda_grid:
            model = make_model(CONTROL_STAGE_E, cfg, hidden, vocab, rank, seed, sample, teacher)
            train_stage_e(model, teacher, train_chunks, dev_chunks, lam, cfg, epochs=epochs,
                          seed=seed)
            dev_results.append({"control": CONTROL_STAGE_E, "rank": rank, "lambda": lam,
                                "split": "dev",
                                "metrics": evaluate(model, teacher, dev_chunks, cfg)})
    selection = select_on_dev(dev_results, cfg.selection_kl_tolerance)
    rank, lam = selection["rank"], selection["lambda"]

    held_out, frozen = {}, None
    for control in (CONTROL_STAGE_E, CONTROL_NPR_ONLY, CONTROL_RANDOM, CONTROL_PCA,
                    CONTROL_REDUCED_RANK, CONTROL_FULL_HIDDEN):
        control_lam = lam if control == CONTROL_STAGE_E else 0.0
        model = make_model(control, cfg, hidden, vocab, rank, seed, sample, teacher)
        info = train_stage_e(model, teacher, train_chunks, dev_chunks, control_lam, cfg,
                             epochs=epochs, seed=seed)
        held_out[control] = {"lambda": control_lam, "best_epoch": info["best_epoch"],
                             **evaluate(model, teacher, held_chunks, cfg)}
        if control == CONTROL_STAGE_E:
            frozen = model
    for param in frozen.parameters():
        param.requires_grad_(False)
    oracle = [local_fit_oracle(teacher.batch(c, True), frozen, teacher, rank) for c in held_chunks]
    held_out[CONTROL_LOCAL_ORACLE] = {
        key: (sum(o[key] for o in oracle) / len(oracle) if key != "note" else oracle[0][key])
        for key in oracle[0]
    }

    # B. generalization: topic / difficulty 값을 가진 root 전체를 떼어 frozen encoder로 평가합니다.
    generalization = {}
    pool = by_split["dev"] + by_split["held_out"]
    for axis in ("topic", "difficulty"):
        values = sorted({p.meta.get(axis) for p in pool if p.meta.get(axis) is not None}, key=str)
        if len(values) < 2:
            continue
        held_value = {values[-1]}
        split = generalization_split(pool, axis, held_value)
        generalization[f"{axis}={sorted(held_value, key=str)}"] = evaluate(
            frozen, teacher, chunked(_records(split["held_out"]), batch_size), cfg)
    from .mp_views import make_mp_views

    view_records = []
    for p in by_split["held_out"]:
        for view in make_mp_views(p.text)[0][: cfg.max_mp_views]:
            record = p.record()
            record.update(problem_id=f"{p.problem_id}#{view.kind}", text=view.text)
            view_records.append(record)
    if view_records:
        generalization["mp_family=format_views"] = evaluate(
            frozen, teacher, chunked(view_records, batch_size), cfg)

    # 짧은 수학 continuation audit (contrast probe 구간의 full-vocabulary KL).
    continuation = {}
    base = _records(by_split["held_out"][:1])[0]
    for family in sorted({p.family for p in make_contrast_probes(seed)}):
        records = []
        for k, probe in enumerate(p for p in make_contrast_probes(seed) if p.family == family):
            for side, text in (("a", probe.text_a), ("b", probe.text_b)):
                record = dict(base)
                record.update(problem_id=f"{base['problem_id']}#{family}{k}{side}", probe_text=text)
                records.append(record)
        batch = teacher.batch(records, need_vjp=False)
        probe_mask = batch.mask & (batch.segments == SEG_PROBE)
        with torch.no_grad():
            z = frozen.encode_state(batch.state_out)[:, -1]
            full = full_vocab_kl(batch.normed[probe_mask], teacher.head_weight, teacher.head_bias,
                                 z[probe_mask], frozen.decoder, frozen.decoder_bias,
                                 cfg.vocab_chunk)
        continuation[family] = float(full.mean())

    pca = pca_basis(sample[0], sample[1], rank)
    held_for_c = teacher.batch(_records(by_split["held_out"][:intervention_limit]), True)
    intervention = [intervention_recovery(frozen, held_for_c, teacher, g, seed, pca)
                    for g in range(cfg.n_macro)]

    meta = StageECheckpointMeta(
        control=CONTROL_STAGE_E, rank=rank, n_macro=cfg.n_macro, hidden_size=hidden,
        vocab_size=vocab, sensitivity_lambda=lam, npr_topk=teacher.topk,
        sketch={"name": "logit_contrast_sketch", "q": cfg.sketch_q, "seed": cfg.sketch_seed,
                "audit_seed": cfg.audit_sketch_seed},
        native=native_meta, dataset=dataset_meta, selection=selection,
        prompt_policy=teacher.policy.as_dict(), notes=[status],
    )
    save_stage_e(frozen, meta, run_dir / "stage_e")
    report = {
        "schema": EXPERIMENT_SCHEMA,
        "status": status,
        "selection": selection,
        "dev_grid": dev_results,
        "A_held_out_fidelity": held_out,
        "B_generalization": generalization,
        "continuation_audit_full_kl": continuation,
        "C_intervention": intervention,
        "reporting": ("native fidelity (A/B), causal use (C) and official robustness prediction "
                      "are reported separately"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (run_dir / "stage_e_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return report


def run_toy_experiment(cfg: StageEConfig, run_dir: str | Path, *, epochs: int = 60,
                       seed: int = 0) -> dict:
    """CPU toy에서 전체 pipeline을 한 번 돌립니다. 결과는 연구 성능이 아닙니다."""
    from .native_toy import ToyTokenizer, build_toy_hybrid

    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = build_toy_hybrid(hidden_size=16, n_layers=8, vocab_size=48, seed=seed)
    tokenizer = ToyTokenizer(vocab_size=48)
    teacher = NativeTeacher(target, tokenizer, cfg, PromptPolicy(max_tokens=cfg.max_tokens), topk=8)
    problems = assign_splits(toy_problems(seed=seed), seed=seed)
    by_split = {name: [p for p in problems if p.split == name]
                for name in ("train", "dev", "held_out")}
    return run_discovery(
        teacher, by_split, cfg, run_dir, epochs=epochs, seed=seed, batch_size=64,
        native_meta={"model_id": "aimo/toy-hybrid", "model_revision": "toy",
                     "layout": "pre_norm", "macro_boundaries": teacher.bounds},
        dataset_meta={"source": "aimo/toy", "split_seed": seed,
                      "n_by_split": {k: len(v) for k, v in by_split.items()}},
        status=STATUS_TOY,
    )
