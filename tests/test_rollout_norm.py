"""Rollout raw recurrence와 normalization test."""

from __future__ import annotations

import copy

import pytest
import torch

from aimo.config import config_from_dict
from aimo.data import collate, compute_norm_stats, sample_pairs
from aimo.model import build_model
from aimo.rollout import rollout
from helpers import tiny_payload


def make(name: str, datasets):
    cfg = config_from_dict(tiny_payload(model={"name": name}))
    train = datasets["train"]
    return build_model(cfg, train.hidden_size, train.n_blocks, train.n_landmarks)


def test_persistence_rollout_matches_hand_computed_raw_recurrence(datasets, stats, batch):
    """V_hat=0이면 U_hat = U_original이고 state는 raw space에서만 누적됩니다."""
    model = make("persistence", datasets)
    result = rollout(model, batch.input_a, stats, horizon=2)
    inp = batch.input_a
    expected = inp.var_state_prefix[:, batch.cut] + inp.orig_updates[:, batch.cut].sum(dim=-2)
    assert torch.allclose(result.steps[0].state_hat_next, expected, atol=1e-6)
    # 두 번째 step은 첫 step의 예측 state를 이어 씁니다.
    expected2 = expected + inp.orig_updates[:, batch.cut + 1].sum(dim=-2)
    assert torch.allclose(result.steps[1].state_hat_next, expected2, atol=1e-6)


def test_rollout_difference_recurrence_identity(datasets, stats, batch):
    """valid landmark에서 D_hat[d+1] = D_hat[d] + sum_c V_hat[d] 가 성립합니다.

    invalid landmark는 original 쪽 residual identity 자체가 깨져 있으므로 (padding junk)
    계약대로 valid landmark에서만 확인합니다.
    """
    model = make("loop4", datasets)
    model.eval()
    with torch.no_grad():
        result = rollout(model, batch.input_a, stats, horizon=3)
    inp = batch.input_a
    valid = batch.valid.unsqueeze(-1)  # [B, P, 1]
    d_prev = inp.var_state_prefix[:, batch.cut] - inp.orig_state[:, batch.cut]
    for step in result.steps:
        expected = d_prev + step.v_hat_raw.sum(dim=-2)
        diff = (step.state_diff_hat_next - expected).abs() * valid
        assert float(diff.max()) < 1e-4
        d_prev = step.state_diff_hat_next


def test_rollout_does_not_reuse_real_variant_future(datasets, stats):
    model = make("loop4", datasets)
    model.eval()
    cut = 1

    def run(source):
        generator = torch.Generator().manual_seed(0)
        samples = sample_pairs(source["train"], generator, 1)
        for sample in samples:
            sample.cut = cut
        with torch.no_grad():
            return rollout(model, collate(samples).input_a, stats, horizon=4)

    corrupted = copy.deepcopy(datasets)
    for group in corrupted["train"].groups:
        for variant in group.variants:
            variant.state[cut + 1 :] += 11.0
            variant.updates[cut:] *= -3.0
    clean, dirty = run(datasets), run(corrupted)
    assert clean.n_steps == dirty.n_steps
    for a, b in zip(clean.steps, dirty.steps, strict=True):
        assert torch.equal(a.v_hat_raw, b.v_hat_raw)
        assert torch.equal(a.state_hat_next, b.state_hat_next)


def test_rollout_stops_at_the_layer_boundary(datasets, stats, bucketed):
    model = make("persistence", datasets)
    n_blocks = datasets["train"].n_blocks
    samples = bucketed[0]
    for sample in samples:
        sample.cut = n_blocks - 1  # 마지막 block에서 시작하면 한 step만 가능합니다.
    batch = collate(samples)
    result = rollout(model, batch.input_a, stats, horizon=4)
    assert result.n_steps == 1
    assert result.steps[-1].depth == n_blocks - 1
    assert result.state_diff_at(n_blocks + 1) is None


def test_normalization_inverse_roundtrip(stats, batch):
    target = batch.target_a
    roundtrip = stats.denorm_target(stats.norm_target(target, batch.cut), batch.cut)
    active = stats.target_active[batch.cut].view(1, 1, 2, 1).expand_as(target)
    assert torch.allclose(roundtrip[active], target[active], atol=1e-6)


def test_normalization_uses_train_originals_only_and_ignores_invalid_landmarks():
    """invalid landmark와 variant 값을 바꿔도 train originals scale은 그대로입니다."""
    cfg = config_from_dict(
        tiny_payload(data={"synthetic": {"n_landmarks": 4, "invalid_landmark_prob": 0.9}})
    )
    from aimo.data import make_synthetic_dataset

    datasets = make_synthetic_dataset(cfg)
    before = compute_norm_stats(datasets["train"])
    corrupted = copy.deepcopy(datasets)
    n_invalid = 0
    for group in corrupted["train"].groups:
        page = group.original
        invalid = ~page.valid
        n_invalid += int(invalid.sum())
        page.state[:, invalid] *= 50.0
        page.updates[:, invalid] *= 50.0
        for variant in group.variants:  # variant 통계는 애초에 쓰지 않습니다.
            variant.state *= 9.0
            variant.updates *= 9.0
    assert n_invalid > 0, "invalid landmark가 있는 자료여야 합니다"
    after = compute_norm_stats(corrupted["train"])
    assert before.hash() == after.hash()
    assert before.source == "train_originals"


def test_zero_scale_is_marked_inactive_and_never_divides_by_zero(datasets):
    stats = compute_norm_stats(datasets["train"])
    inactive = ~stats.target_active
    assert bool(inactive.any()), "zero-scale 대조 stream이 있어야 합니다"
    depth, stream = (int(x) for x in torch.nonzero(inactive)[0])
    assert float(stats.target_scale[depth, stream]) == 0.0
    probe = torch.ones(2, 3, 2, 8)  # [B, P, 2, H]
    normalized = stats.norm_target(probe, depth)
    assert torch.isfinite(normalized).all()
    # floor로 clamp되므로 값이 폭발하지 않습니다.
    assert float(normalized[..., stream, :].abs().max()) == pytest.approx(1.0 / stats.floor)


def test_scale_is_recomputed_inside_each_training_subset(datasets):
    """25/50/100% subset마다 normalization을 그 subset 안에서만 계산합니다."""
    full = datasets["train"]
    half = full.subset(0.5)
    assert len(half.groups) < len(full.groups)
    assert half.groups == full.groups[: len(half.groups)]  # nested prefix
    assert compute_norm_stats(half).hash() != compute_norm_stats(full).hash()
    quarter = full.subset(0.25)
    assert quarter.groups == half.groups[: len(quarter.groups)]  # nested subset
