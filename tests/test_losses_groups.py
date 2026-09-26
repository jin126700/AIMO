"""Sibling / M0 / support-swap의 group semantics와 loss 항 test."""

from __future__ import annotations

import pytest
import torch

from aimo.config import config_from_dict
from aimo.data import (
    PairSample,
    collate,
    enumerate_pairs,
    sample_pairs,
    swap_support,
)
from aimo.losses import compute_loss, next_update_loss, rollout_loss, within_original_loss
from aimo.model import build_model
from helpers import tiny_payload


def make(name: str, datasets):
    cfg = config_from_dict(tiny_payload(model={"name": name}))
    train = datasets["train"]
    return build_model(cfg, train.hidden_size, train.n_blocks, train.n_landmarks)


def test_sibling_pairs_always_use_two_distinct_variants(datasets):
    generator = torch.Generator().manual_seed(3)
    samples = sample_pairs(datasets["train"], generator, 4)
    assert samples
    for sample in samples:
        assert sample.index_b is not None
        assert sample.index_a != sample.index_b
        page_a = sample.group.variants[sample.index_a]
        page_b = sample.group.variants[sample.index_b]
        assert page_a.variant_id != page_b.variant_id
        # identity variant를 서로 다른 sibling 두 개로 세지 않습니다.
        assert not (page_a.is_identity and page_b.is_identity)


def test_collate_rejects_a_pair_of_the_same_variant(datasets):
    group = datasets["train"].groups[0]
    bad = PairSample(group=group, index_a=0, index_b=0, cut=0)
    with pytest.raises(ValueError, match="distinct variant ids"):
        collate([bad])


def test_collate_requires_one_shared_cut(datasets):
    group = datasets["train"].groups[0]
    with pytest.raises(ValueError, match="single shared cut"):
        collate(
            [
                PairSample(group, 0, 1, cut=0),
                PairSample(group, 0, 1, cut=1),
            ]
        )


def test_collate_uses_common_valid_landmarks(datasets):
    group = datasets["train"].groups[0]
    batch = collate([PairSample(group, 0, 1, cut=1)])
    expected = (
        group.original.valid & group.variants[0].valid & group.variants[1].valid
    )
    assert torch.equal(batch.valid[0], expected)


def test_identity_example_has_zero_target(datasets):
    groups = [g for g in datasets["train"].groups if any(v.is_identity for v in g.variants)]
    assert groups
    group = groups[0]
    index = next(i for i, v in enumerate(group.variants) if v.is_identity)
    other = next(i for i in range(len(group.variants)) if i != index)
    batch = collate([PairSample(group, index, other, cut=0)])
    assert torch.count_nonzero(batch.target_a) == 0
    assert bool(batch.is_identity_a[0])


def test_within_loss_only_counts_originals_that_have_a_sibling(datasets, stats):
    group = datasets["train"].groups[0]
    batch = collate([PairSample(group, 0, None, cut=1)])
    assert not bool(batch.has_sibling[0])
    zeros = torch.zeros_like(batch.target_a)
    term = within_original_loss(zeros, zeros, batch, stats)
    assert not term.defined


def test_within_loss_is_zero_when_predicted_difference_matches(datasets, stats):
    group = datasets["train"].groups[0]
    batch = collate([PairSample(group, 0, 1, cut=1)])
    pred_a = stats.norm_target(batch.target_a, batch.cut)
    pred_b = stats.norm_target(batch.target_b, batch.cut)
    term = within_original_loss(pred_a, pred_b, batch, stats)
    assert term.defined
    assert float(term.value) == pytest.approx(0.0, abs=1e-10)


def test_next_loss_is_zero_for_a_perfect_prediction(datasets, stats, batch):
    pred_a = stats.norm_target(batch.target_a, batch.cut)
    pred_b = stats.norm_target(batch.target_b, batch.cut)
    term = next_update_loss(pred_a, pred_b, batch, stats)
    assert float(term.value) == pytest.approx(0.0, abs=1e-10)


def test_each_original_gets_equal_weight(datasets, stats):
    """batch 평균은 원문별 loss의 단순 평균과 같습니다."""
    model = make("loop4", datasets)
    model.eval()
    groups = datasets["train"].groups[:2]
    singles = []
    with torch.no_grad():
        for group in groups:
            one = collate([PairSample(group, 0, 1, cut=1)])
            pred_a = model(one.input_a, stats)
            pred_b = model(one.input_b, stats)
            singles.append(float(next_update_loss(pred_a, pred_b, one, stats).value))
        both = collate([PairSample(g, 0, 1, cut=1) for g in groups])
        term = next_update_loss(model(both.input_a, stats), model(both.input_b, stats), both, stats)
    assert float(term.value) == pytest.approx(sum(singles) / len(singles), rel=1e-5)


def test_support_swap_stays_inside_the_same_original(datasets, stats):
    group = datasets["train"].groups[0]
    batch = collate([PairSample(group, 0, 1, cut=1)])
    swapped = swap_support(batch, torch.Generator().manual_seed(0))
    assert swapped.original_ids == batch.original_ids
    # original 참조와 target은 그대로, 관측 prefix만 sibling의 것으로 바뀝니다.
    assert torch.equal(swapped.input_a.orig_state, batch.input_a.orig_state)
    assert torch.equal(swapped.target_a, batch.target_a)
    assert torch.equal(swapped.input_a.var_state_prefix, batch.input_b.var_state_prefix)
    assert not torch.equal(swapped.input_a.var_state_prefix, batch.input_a.var_state_prefix)


def test_support_swap_changes_loop_prediction_but_not_m0(datasets, stats, batch):
    swapped = swap_support(batch, torch.Generator().manual_seed(0))
    loop = make("loop4", datasets)
    loop.eval()
    m0 = make("m0", datasets)
    m0.eval()
    with torch.no_grad():
        assert not torch.equal(loop(batch.input_a, stats), loop(swapped.input_a, stats))
        assert torch.equal(m0(batch.input_a, stats), m0(swapped.input_a, stats))


def test_out_of_range_horizon_is_undefined(datasets, stats):
    model = make("persistence", datasets)
    group = datasets["train"].groups[0]
    n_blocks = datasets["train"].n_blocks
    batch = collate([PairSample(group, 0, 1, cut=n_blocks - 1)])
    term, per_horizon = rollout_loss(model, batch, stats, horizons=(2, 4))
    assert not term.defined
    assert not per_horizon[2].defined
    assert not per_horizon[4].defined


def test_total_loss_uses_the_configured_weights(datasets, stats, batch):
    model = make("loop4", datasets)
    model.eval()
    with torch.no_grad():
        breakdown = compute_loss(model, batch, stats, w_next=1.0, w_within=1.0, w_roll=0.25)
    expected = (
        float(breakdown.next_update.value)
        + float(breakdown.within.value)
        + 0.25 * float(breakdown.roll.value)
    )
    assert float(breakdown.total) == pytest.approx(expected, rel=1e-6)


def test_enumerate_pairs_is_deterministic(datasets):
    first = enumerate_pairs(datasets["validation"], [0, 1])
    second = enumerate_pairs(datasets["validation"], [0, 1])
    assert [(s.group.original_id, s.index_a, s.index_b, s.cut) for s in first] == [
        (s.group.original_id, s.index_a, s.index_b, s.cut) for s in second
    ]
