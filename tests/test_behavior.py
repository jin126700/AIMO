"""Behavior view test: shared core, 정보 경로 분리, panel 처리, untrained head."""

from __future__ import annotations

import copy

import pytest
import torch

from aimo.behavior import AimoModel, build_behavior_layout
from aimo.config import config_from_dict
from aimo.data import (
    PairSample,
    build_behavior_input,
    collate,
    collate_panels,
    panel_members,
    swap_panel_support,
)
from aimo.model import KIND_OBS, KIND_PAIR, KIND_REF, build_attention_mask, build_model
from helpers import tiny_payload


def make(name: str, datasets):
    cfg = config_from_dict(tiny_payload(model={"name": name}))
    train = datasets["train"]
    return build_model(cfg, train.hidden_size, train.n_blocks, train.n_landmarks)


@pytest.fixture
def panel_batch(datasets):
    return collate_panels(datasets["train"].groups[:4])


# --------------------------------------------------------------------------------------
# shared parameter identity
# --------------------------------------------------------------------------------------


def test_behavior_and_flow_share_the_same_core_object(datasets, stats, panel_batch):
    model = make("joint", datasets)
    assert isinstance(model, AimoModel)
    assert model.core is model.flow.core
    # 같은 parameter 객체를 쓰므로 core parameter id 집합이 완전히 겹칩니다.
    core_ids = {id(p) for p in model.core.parameters()}
    flow_ids = {id(p) for p in model.flow.parameters()}
    assert core_ids <= flow_ids
    # behavior head는 core parameter를 복제하지 않습니다.
    head_ids = {id(p) for p in model.behavior.parameters()}
    assert not (head_ids & core_ids)
    # parameter() 는 공유 parameter를 중복 반환하지 않습니다.
    all_ids = [id(p) for p in model.parameters()]
    assert len(all_ids) == len(set(all_ids))


def test_shared_block_is_called_n_loops_times_in_both_views(datasets, stats, panel_batch):
    model = make("joint", datasets)
    model.eval()
    calls: list[int] = []
    handle = model.core.blocks[0].register_forward_hook(lambda *_: calls.append(1))
    model.forward_behavior(panel_batch.inputs, stats)
    behavior_calls = len(calls)
    calls.clear()
    group = datasets["train"].groups[0]
    model(collate([PairSample(group, 0, 1, cut=1)]).input_a, stats)
    flow_calls = len(calls)
    handle.remove()
    assert behavior_calls == flow_calls == model.core.n_loops == 4


def test_core_gradient_accumulates_from_both_views(datasets, stats, panel_batch):
    from aimo.losses import behavior_loss, compute_loss, joint_loss

    model = make("joint", datasets)
    model.train()
    group = datasets["train"].groups[0]
    flow_batch = collate([PairSample(group, 0, 1, cut=1)])
    breakdown = joint_loss(
        behavior_loss(model, panel_batch, stats), compute_loss(model, flow_batch, stats)
    )
    breakdown.total.backward()
    core_grad = sum(
        float(p.grad.abs().sum()) for p in model.core.parameters() if p.grad is not None
    )
    behavior_grad = sum(
        float(p.grad.abs().sum()) for p in model.behavior.parameters() if p.grad is not None
    )
    flow_grad = float(model.flow.readout.weight.grad.abs().sum())
    assert core_grad > 0 and behavior_grad > 0 and flow_grad > 0


# --------------------------------------------------------------------------------------
# 정보 경로 분리와 leakage
# --------------------------------------------------------------------------------------


def test_behavior_reads_every_variant_depth(datasets):
    n_blocks, n_p = datasets["train"].n_blocks, datasets["train"].n_landmarks
    layout = build_behavior_layout(n_blocks, n_p, torch.device("cpu"))
    variant_depths = layout.depth[layout.kind == KIND_OBS].unique().tolist()
    # variant는 state[0:L+1] 전체를 읽습니다 (마지막 block을 누락하지 않습니다).
    assert variant_depths == list(range(n_blocks + 1))
    assert layout.cut == n_blocks
    assert int((layout.kind == KIND_PAIR).sum()) == 1


def test_behavior_mask_keeps_the_same_leakage_rules(datasets):
    n_blocks, n_p = 4, 3
    layout = build_behavior_layout(n_blocks, n_p, torch.device("cpu"))
    cell_valid = torch.ones(1, layout.kind.shape[0], dtype=torch.bool)
    allowed = ~build_attention_mask(layout, cell_valid)[0]
    ref_rows = layout.kind == KIND_REF
    # original reference는 original만 읽습니다.
    assert not bool(allowed[ref_rows][:, layout.kind != KIND_REF].any())
    # variant depth r은 variant depth <= r만 읽습니다.
    for i in torch.nonzero(layout.kind == KIND_OBS).flatten().tolist():
        cols = torch.nonzero(allowed[i]).flatten()
        kinds = layout.kind[cols]
        assert not bool((kinds == KIND_PAIR).any())
        later = (kinds == KIND_OBS) & (layout.depth[cols] > layout.depth[i])
        assert not bool(later.any())
    # pair readout은 original/variant 전체를 읽습니다.
    query = int(torch.nonzero(layout.kind == KIND_PAIR)[0])
    cols = torch.nonzero(allowed[query]).flatten()
    assert int((layout.kind[cols] == KIND_REF).sum()) == (n_blocks + 1) * n_p
    assert int((layout.kind[cols] == KIND_OBS).sum()) == (n_blocks + 1) * n_p


def test_flow_prediction_is_unchanged_after_a_behavior_forward(datasets, stats, panel_batch):
    """full-page behavior 경로를 지나도 flow에 variant future가 새지 않습니다."""
    model = make("joint", datasets)
    model.eval()
    group = datasets["train"].groups[0]
    flow_batch = collate([PairSample(group, 0, 1, cut=1)])
    with torch.no_grad():
        before = model(flow_batch.input_a, stats)
        model.forward_behavior(panel_batch.inputs, stats)  # 전체 Page를 본 뒤
        after = model(flow_batch.input_a, stats)
    assert torch.equal(before, after)


def test_flow_still_ignores_variant_future_in_the_joint_model(datasets, stats):
    model = make("joint", datasets)
    model.eval()
    cut = 1

    def predict(source):
        group = source["train"].groups[0]
        return model(collate([PairSample(group, 0, 1, cut=cut)]).input_a, stats)

    corrupted = copy.deepcopy(datasets)
    for group in corrupted["train"].groups:
        for variant in group.variants:
            variant.state[cut + 1 :] += 9.0
            variant.updates[cut:] -= 4.0
    with torch.no_grad():
        assert torch.equal(predict(datasets), predict(corrupted))


def test_behavior_input_carries_no_label_or_id(datasets):
    group = datasets["train"].groups[0]
    inputs = build_behavior_input([(group.original, group.variants[1])])
    fields = set(inputs.__dict__)
    for forbidden in ("label", "drop", "robust", "variant_id", "original_id", "counts", "topic"):
        assert not any(forbidden in name for name in fields)
    # original/variant validity가 독립 field로 들어옵니다.
    assert "orig_valid" in fields and "var_valid" in fields


def test_independent_original_and_variant_validity(datasets, stats):
    """variant validity만 바꾸면 결과가 바뀌고, original validity는 별도로 동작합니다."""
    model = make("behavior", datasets)
    model.eval()
    group = datasets["train"].groups[0]
    inputs = build_behavior_input([(group.original, group.variants[1])])
    with torch.no_grad():
        base = model.forward_behavior(inputs, stats).pair_drop.clone()
        # landmark 1의 validity만 한쪽에서 뒤집습니다 (batch 크기는 1).
        flipped = copy.deepcopy(inputs)
        flipped.var_valid[0, 1] = ~flipped.var_valid[0, 1]
        variant_changed = model.forward_behavior(flipped, stats).pair_drop
        flipped_orig = copy.deepcopy(inputs)
        flipped_orig.orig_valid[0, 1] = ~flipped_orig.orig_valid[0, 1]
        original_changed = model.forward_behavior(flipped_orig, stats).pair_drop
    assert not torch.equal(base, variant_changed)
    assert not torch.equal(base, original_changed)
    assert not torch.equal(variant_changed, original_changed)


def test_m0_behavior_ignores_the_variant_page(datasets, stats, panel_batch):
    model = make("behavior_m0", datasets)
    model.eval()
    with torch.no_grad():
        out = model.forward_behavior(panel_batch.inputs, stats)
        panel = model.panel_outputs(
            out, panel_batch.pair_panel, panel_batch.pair_slot, panel_batch.panel_mask
        )
    # 같은 original의 모든 variant가 같은 예측을 받습니다 (variant 정보가 없으므로).
    for b in range(panel_batch.n_panels):
        slots = torch.nonzero(panel_batch.panel_mask[b]).flatten().tolist()
        values = [float(panel.pair_drop[b, slot]) for slot in slots]
        assert max(values) - min(values) < 1e-6
    # variant Page를 무작위로 바꿔도 결과가 같습니다.
    noisy = copy.deepcopy(panel_batch.inputs)
    noisy.var_state = torch.randn_like(noisy.var_state)
    noisy.var_updates = torch.randn_like(noisy.var_updates)
    with torch.no_grad():
        assert torch.equal(out.pair_drop, model.forward_behavior(noisy, stats).pair_drop)


def test_m0_sequence_length_does_not_depend_on_variant_count(datasets, stats):
    """M0에 variant 수/길이 정보가 새지 않습니다."""
    model = make("behavior_m0", datasets)
    model.eval()
    small = collate_panels(datasets["train"].groups[:1])
    with torch.no_grad():
        one = model.forward_behavior(small.inputs, stats).pair_drop
    assert float(one.max() - one.min()) < 1e-6


# --------------------------------------------------------------------------------------
# panel 처리
# --------------------------------------------------------------------------------------


def test_panel_permutation_does_not_change_predictions(datasets, stats):
    from aimo.data import OriginalGroup

    model = make("behavior", datasets)
    model.eval()
    groups = datasets["train"].groups[:3]
    flipped = [
        OriginalGroup(
            original_id=g.original_id,
            original=g.original,
            variants=list(reversed(g.variants)),
            panel_id=g.panel_id,
            pair_labels=g.pair_labels,
            panel_label=g.panel_label,
            metadata=g.metadata,
        )
        for g in groups
    ]
    with torch.no_grad():
        a = collate_panels(groups)
        b = collate_panels(flipped)
        pa = model.panel_outputs(
            model.forward_behavior(a.inputs, stats), a.pair_panel, a.pair_slot, a.panel_mask
        )
        pb = model.panel_outputs(
            model.forward_behavior(b.inputs, stats), b.pair_panel, b.pair_slot, b.panel_mask
        )
    assert torch.allclose(pa.robust_prob, pb.robust_prob, atol=1e-6)
    assert torch.allclose(pa.max_drop, pb.max_drop, atol=1e-6)


def test_partial_and_padded_panels_are_handled(datasets, stats):
    from aimo.data import OriginalGroup

    model = make("behavior", datasets)
    model.eval()
    full = datasets["train"].groups[1]
    partial = OriginalGroup(
        original_id="partial",
        original=full.original,
        variants=[full.variants[-1]],
        panel_id="partial#panel",
        pair_labels={},
        panel_label=None,
    )
    batch = collate_panels([full, partial])
    assert batch.panel_mask[1].tolist().count(True) == 1  # padding slot이 생깁니다
    with torch.no_grad():
        panel = model.panel_outputs(
            model.forward_behavior(batch.inputs, stats),
            batch.pair_panel,
            batch.pair_slot,
            batch.panel_mask,
        )
    assert bool(panel.panel_valid.all())
    # padding slot의 pair drop은 NaN이고 pooling weight는 0입니다.
    assert torch.isnan(panel.pair_drop[1, 1])
    assert float(panel.pooling_weights[1, 1]) == 0.0
    assert torch.isfinite(panel.robust_prob).all()


def test_identity_variant_is_not_a_panel_member(datasets):
    groups = [g for g in datasets["train"].groups if any(v.is_identity for v in g.variants)]
    assert groups
    members = panel_members(groups[0])
    assert all(not variant.is_identity for variant in members)
    assert len(members) == len(groups[0].variants) - 1


def test_pair_support_swap_keeps_the_target_and_changes_the_observation(datasets, stats):
    model = make("behavior", datasets)
    model.eval()
    batch = collate_panels(datasets["train"].groups[1:4])
    swapped = swap_panel_support(batch, target_slot=0)
    # target과 original 참조는 그대로입니다 (같은 tensor를 그대로 넘깁니다).
    assert swapped.drop_target is batch.drop_target
    assert swapped.robust_target is batch.robust_target
    assert torch.equal(swapped.inputs.orig_state, batch.inputs.orig_state)
    # slot 0의 variant 관측만 sibling의 것으로 바뀝니다.
    first = int(torch.nonzero(batch.pair_slot == 0)[0])
    assert not torch.equal(swapped.inputs.var_state[first], batch.inputs.var_state[first])
    with torch.no_grad():
        before = model.forward_behavior(batch.inputs, stats).pair_drop[first]
        after = model.forward_behavior(swapped.inputs, stats).pair_drop[first]
    assert not torch.equal(before, after)


def test_untrained_heads_are_reported(datasets):
    model = make("joint", datasets)
    assert model.trained_heads() == {"pair_drop": False, "robust": False, "max_drop": False}
    model.mark_trained("pair_drop")
    assert model.trained_heads()["pair_drop"] is True
    assert model.trained_heads()["robust"] is False


def test_pair_embedding_dimension_is_d_model(datasets, stats, panel_batch):
    model = make("joint", datasets)
    out = model.forward_behavior(panel_batch.inputs, stats)
    assert out.z.shape == (panel_batch.inputs.batch_size, model.core.d_model)
    assert bool((out.pair_drop.abs() <= 1.0).all())  # tanh 범위


def test_behavior_baselines_have_no_core(datasets, stats, panel_batch):
    for name in ("constant", "raw_change"):
        model = make(name, datasets)
        report = model.param_report()
        assert report.core == 0
        out = model.forward_behavior(panel_batch.inputs, stats)
        panel = model.panel_outputs(
            out, panel_batch.pair_panel, panel_batch.pair_slot, panel_batch.panel_mask
        )
        assert out.pair_drop.shape == (panel_batch.inputs.batch_size,)
        assert torch.isfinite(panel.robust_logit).all()
        with pytest.raises(NotImplementedError):
            model.forward_flow(None, stats)


def test_constant_baseline_gives_the_same_drop_everywhere(datasets, stats, panel_batch):
    model = make("constant", datasets)
    out = model.forward_behavior(panel_batch.inputs, stats)
    assert float(out.pair_drop.max() - out.pair_drop.min()) == 0.0
