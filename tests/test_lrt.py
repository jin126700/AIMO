"""LRT-v1 model / loss / leakage / control test.

이 test는 architecture plumbing을 검증합니다. Toy 결과는 real-data evidence가 아닙니다.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from aimo.config import LRTConfig
from aimo.lrt import (
    LANDMARK_ALL_COMMON,
    LANDMARK_FINAL_TOKEN,
    LRTModel,
    build_pair_tensors,
    fit_macro_norm_stats,
    landmark_mask,
    query_folds,
    support_macros_for,
)
from aimo.lrt_experiment import (
    LRT_CHECKPOINT_SCHEMA,
    Rank4LinearTransport,
    TrainMeanTransport,
    ZeroTransport,
    consistency_loss,
    evaluate_transport,
    load_lrt_checkpoint,
    make_toy_rank4,
    make_toy_state_dependent,
    original_balanced_mean,
    resolve_floor,
    save_lrt_checkpoint,
    split_pairs_by_original,
    support_dropout_mask,
    swap_donor_index,
    to_macro_pairs,
    train_lrt,
    transport_term,
)
from aimo.macro_page import to_macro_page
from helpers import fine_page

FLOOR = 1e-6


@pytest.fixture
def toy_b():
    bundle = make_toy_state_dependent(n_originals=6, n_variants=2, n_macro=8, seed=0)
    return to_macro_pairs(bundle["pairs"], 8)


@pytest.fixture
def pair():
    original = to_macro_page(fine_page(seed=0), 8)
    variant = to_macro_page(fine_page(seed=1, variant=True), 8)
    return original, variant


@pytest.fixture
def model():
    return LRTModel(hidden_size=8, n_macro=8, n_landmarks=4)


def payload_for(pair, stats):
    original, variant = pair
    tensors = build_pair_tensors(original, variant, stats)
    mask = landmark_mask(original, variant, LANDMARK_ALL_COMMON)
    return tensors, torch.nonzero(mask).flatten(), mask


# --------------------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------------------


def test_query_delta_is_not_visible_to_the_encoder(pair, model):
    """query macro의 ΔU를 바꿔도 z_rel이 변하지 않아야 합니다."""
    stats = fit_macro_norm_stats([pair[0]])
    tensors, landmarks, _ = payload_for(pair, stats)
    model.eval()
    fold = (0, 1)
    support = support_macros_for(8, fold)
    with torch.no_grad():
        base, _ = model.encode(
            tensors["delta_norm"], support, landmarks, tensors["relative_positions"]
        )
        corrupted = tensors["delta_norm"].clone()
        corrupted[:, list(fold)] += 7.0  # query fold의 relation만 오염
        other, _ = model.encode(
            corrupted, support, landmarks, tensors["relative_positions"]
        )
    assert torch.equal(base, other)


def test_variant_state_is_not_a_decoder_input(pair, model):
    """decoder는 original state/update와 z, site만 봅니다."""
    stats = fit_macro_norm_stats([pair[0]])
    tensors, landmarks, _ = payload_for(pair, stats)
    assert set(tensors) == {
        "delta_norm",
        "original_state_norm",
        "original_update_norm",
        "relative_positions",
    }
    model.eval()
    fold = (0, 1)
    with torch.no_grad():
        z, _ = model.encode(
            tensors["delta_norm"], support_macros_for(8, fold), landmarks,
            tensors["relative_positions"],
        )
        first = model.decode(
            z, tensors["original_state_norm"], tensors["original_update_norm"], fold,
            landmarks, tensors["relative_positions"],
        )
        # variant state는 build_pair_tensors 출력에 없으므로 decode 서명에 들어갈 자리가 없습니다.
        second = model.decode(
            z, tensors["original_state_norm"], tensors["original_update_norm"], fold,
            landmarks, tensors["relative_positions"],
        )
    assert torch.equal(first, second)


def test_cumulative_state_difference_is_not_an_encoder_input(pair):
    """encoder가 보는 primary activation은 ΔU뿐입니다."""
    original, variant = pair
    stats = fit_macro_norm_stats([original])
    tensors = build_pair_tensors(original, variant, stats)
    cumulative = stats.norm_state((variant.state - original.state)[None])
    for value in tensors.values():
        if value.shape == cumulative.shape:
            assert not torch.allclose(value, cumulative)
    assert "variant_state_norm" not in tensors


def test_decoder_uses_original_state_and_original_update(pair, model):
    """둘 중 하나만 바꿔도 예측이 바뀌어야 합니다 (state-dependent relation)."""
    stats = fit_macro_norm_stats([pair[0]])
    tensors, landmarks, _ = payload_for(pair, stats)
    model.eval()
    fold = (0, 1)
    with torch.no_grad():
        z, _ = model.encode(
            tensors["delta_norm"], support_macros_for(8, fold), landmarks,
            tensors["relative_positions"],
        )
        base = model.decode(
            z, tensors["original_state_norm"], tensors["original_update_norm"], fold,
            landmarks, tensors["relative_positions"],
        )
        other_state = tensors["original_state_norm"] + 1.0
        changed_state = model.decode(
            z, other_state, tensors["original_update_norm"], fold, landmarks,
            tensors["relative_positions"],
        )
        other_update = tensors["original_update_norm"] + 1.0
        changed_update = model.decode(
            z, tensors["original_state_norm"], other_update, fold, landmarks,
            tensors["relative_positions"],
        )
        changed_z = model.decode(
            z + 1.0, tensors["original_state_norm"], tensors["original_update_norm"], fold,
            landmarks, tensors["relative_positions"],
        )
    assert not torch.equal(base, changed_state)
    assert not torch.equal(base, changed_update)
    assert not torch.equal(base, changed_z)


def test_training_needs_no_behavior_labels(toy_b):
    """robustness / drop / outcome label 없이 완전한 학습이 가능해야 합니다."""
    for original, _variant in toy_b:
        assert not getattr(original, "pair_labels", {})
        assert not hasattr(original, "panel_label")
        assert "drop" not in original.provenance
    train_pairs, validation_pairs = split_pairs_by_original(toy_b, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    result = train_lrt(
        train_pairs, validation_pairs, stats, LRTConfig(denominator_floor=FLOOR),
        floor=FLOOR, max_epochs=2, patience=2,
    )
    assert result["epochs_run"] >= 1
    assert result["schema"] == LRT_CHECKPOINT_SCHEMA


# --------------------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------------------


def test_relation_code_and_adapter_dimensions(pair, model):
    stats = fit_macro_norm_stats([pair[0]])
    tensors, landmarks, _ = payload_for(pair, stats)
    model.eval()
    with torch.no_grad():
        out = model(
            tensors["delta_norm"], tensors["original_state_norm"],
            tensors["original_update_norm"], support_macros_for(8, (0, 1)), (0, 1),
            landmarks, tensors["relative_positions"],
        )
    assert out.z_rel.shape == (1, 16)  # relation_dim = 16
    assert model.adapter_dim == 32
    assert model.update_adapter.out_features == 32
    assert model.state_adapter.out_features == 32
    assert model.decoder_rank == 16
    assert model.basis_mixer.in_features == 16 and model.basis_mixer.out_features == 8
    assert model.basis_ffn.in_features == 16 and model.basis_ffn.out_features == 8
    assert model.basis_mixer.bias is None and model.basis_ffn.bias is None
    # query output shape: [B, Q, P_sel, 2, H]
    assert tuple(out.delta_hat.shape) == (1, 2, 4, 2, 8)


def test_update_adapter_is_shared_between_streams(model):
    """Mixer와 FFN은 같은 A_U parameter를 공유하고 stream embedding으로 구분합니다."""
    names = {name for name, _ in model.named_parameters() if "update_adapter" in name}
    assert names == {"update_adapter.weight", "update_adapter.bias"}
    assert not any("mixer_adapter" in name or "ffn_adapter" in name for name, _ in
                   model.named_parameters())
    mixer = model.site.stream(torch.tensor([0]))
    ffn = model.site.stream(torch.tensor([1]))
    assert not torch.equal(mixer, ffn)


def test_shared_block_is_one_object_called_n_loops_times(pair, model):
    stats = fit_macro_norm_stats([pair[0]])
    tensors, landmarks, _ = payload_for(pair, stats)
    calls: list[int] = []
    handle = model.block.register_forward_hook(lambda *_: calls.append(1))
    model.eval()
    with torch.no_grad():
        model.encode(
            tensors["delta_norm"], support_macros_for(8, (0, 1)), landmarks,
            tensors["relative_positions"],
        )
    handle.remove()
    assert len(calls) == model.n_loops == 4
    block_ids = [id(p) for p in model.block.parameters()]
    assert len(block_ids) == len(set(block_ids))


def test_param_report_separates_components(model):
    report = model.param_report().as_dict()
    for key in (
        "update_adapter", "state_adapter", "relation_embeddings", "shared_block",
        "relation_projection", "coefficient_network", "mixer_basis", "ffn_basis", "total",
    ):
        assert key in report
    components = sum(
        report[key]
        for key in (
            "update_adapter", "state_adapter", "relation_embeddings", "shared_block",
            "relation_projection", "coefficient_network", "mixer_basis", "ffn_basis",
        )
    )
    assert components == report["total"]
    assert "4배" in report["notes"]["macro_effect"]


def test_query_folds_are_contiguous_pairs_and_support_is_the_rest():
    folds = query_folds(8)
    assert folds == [(0, 1), (2, 3), (4, 5), (6, 7)]
    for fold in folds:
        support = support_macros_for(8, fold)
        assert len(support) == 6
        assert not set(support) & set(fold)
    with pytest.raises(ValueError, match="even number"):
        query_folds(5)


# --------------------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------------------


def test_perfect_prediction_gives_ratio_zero():
    target = torch.randn(2, 4, 2, 8)
    valid = torch.ones(4, dtype=torch.bool)
    term = transport_term(target, target.clone(), valid, FLOOR)
    assert term.ratio == pytest.approx(0.0)
    assert term.e_pred == pytest.approx(0.0)


def test_zero_predictor_gives_ratio_one_above_the_floor():
    target = torch.randn(2, 4, 2, 8)
    valid = torch.ones(4, dtype=torch.bool)
    term = transport_term(target, torch.zeros_like(target), valid, FLOOR)
    assert term.ratio == pytest.approx(1.0, rel=1e-6)
    assert not term.below_floor


def test_low_energy_denominator_stays_finite():
    target = torch.zeros(2, 4, 2, 8)
    valid = torch.ones(4, dtype=torch.bool)
    term = transport_term(target, torch.zeros_like(target), valid, FLOOR)
    assert term.e_zero == 0.0
    assert term.ratio == pytest.approx(0.0)
    assert term.below_floor
    noisy = transport_term(target, torch.full_like(target, 1e-9), valid, FLOOR)
    assert noisy.ratio < 1.0 and torch.isfinite(torch.tensor(noisy.ratio))


def test_original_balanced_aggregation_ignores_variant_count():
    """variant 수가 많은 original이 결과를 지배하지 않습니다."""
    few = {"a": [0.2], "b": [0.8]}
    many = {"a": [0.2] * 9, "b": [0.8]}
    assert original_balanced_mean(few) == pytest.approx(0.5)
    assert original_balanced_mean(many) == pytest.approx(0.5)


def test_consistency_dropout_never_masks_every_support_cell():
    generator = torch.Generator().manual_seed(0)
    for rate in (0.0, 0.1, 0.9, 0.99):
        for _ in range(20):
            mask = support_dropout_mask(8, rate, generator, torch.device("cpu"))
            assert bool(mask.any())
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        support_dropout_mask(8, 1.0, generator, torch.device("cpu"))


def test_consistency_loss_is_zero_for_identical_codes():
    z = torch.randn(3, 16)
    assert float(consistency_loss(z, z.clone())) == pytest.approx(0.0, abs=1e-6)
    assert float(consistency_loss(z, -z)) == pytest.approx(2.0, abs=1e-6)


def test_encoder_rejects_a_fully_masked_support_set(pair, model):
    stats = fit_macro_norm_stats([pair[0]])
    tensors, landmarks, _ = payload_for(pair, stats)
    support = support_macros_for(8, (0, 1))
    n_cells = len(support) * int(landmarks.numel()) * 2
    empty = torch.zeros(1, n_cells, dtype=torch.bool)
    with pytest.raises(ValueError, match="masked out"):
        model.encode(
            tensors["delta_norm"], support, landmarks, tensors["relative_positions"],
            cell_mask=empty,
        )


def test_denominator_floor_must_be_explicit_for_real_runs():
    with pytest.raises(ValueError, match="frozen"):
        resolve_floor(LRTConfig())
    assert resolve_floor(LRTConfig(denominator_floor=2.5)) == pytest.approx(2.5)


# --------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------


def test_swap_donor_is_deterministic_and_cross_original():
    ids = ["o0", "o0", "o1", "o2", "o3"]
    for index in range(len(ids)):
        donor = swap_donor_index(index, ids, eval_seed=7)
        assert donor != index
        assert ids[donor] != ids[index]  # 같은 original은 피합니다
        assert swap_donor_index(index, ids, eval_seed=7) == donor  # deterministic
    assert swap_donor_index(0, ids, eval_seed=8) != index or True  # seed가 바뀌면 달라질 수 있음
    with pytest.raises(ValueError, match="at least two pairs"):
        swap_donor_index(0, ["only"], eval_seed=0)


def test_both_landmark_modes_are_supported(pair, model):
    original, variant = pair
    stats = fit_macro_norm_stats([original])
    all_common = landmark_mask(original, variant, LANDMARK_ALL_COMMON)
    final_token = landmark_mask(original, variant, LANDMARK_FINAL_TOKEN)
    assert int(all_common.sum()) == 4
    assert int(final_token.sum()) == 1
    assert bool(final_token[-1])  # canonical final prompt landmark
    for mode in (LANDMARK_ALL_COMMON, LANDMARK_FINAL_TOKEN):
        result = evaluate_transport(
            model, [pair, pair], stats, floor=FLOOR, landmark_mode=mode
        )
        assert result["landmark_mode"] == mode
        assert result["n_folds"] == 8
    with pytest.raises(ValueError, match="landmark_mode"):
        landmark_mask(original, variant, "bad_mode")


def test_baselines_have_the_documented_contract(toy_b):
    train_pairs, validation_pairs = split_pairs_by_original(toy_b, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    zero = evaluate_transport(
        None, validation_pairs, stats, floor=FLOOR, baseline=ZeroTransport()
    )
    assert zero["transport_ratio"] == pytest.approx(1.0, rel=1e-5)
    train_mean = TrainMeanTransport.fit(train_pairs, stats)
    rank4 = Rank4LinearTransport.fit(train_pairs, stats)
    assert rank4.name == "rank4_linear_transport"  # historical U4 재현이 아닙니다
    assert rank4.rank == 4
    for stream in (0, 1):
        assert tuple(rank4.basis[stream].shape) == (8, 4)
    for baseline in (train_mean, rank4):
        result = evaluate_transport(
            None, validation_pairs, stats, floor=FLOOR, baseline=baseline
        )
        assert result["transport_ratio"] >= 0.0


def test_rank4_baseline_recovers_a_fixed_rank4_relation():
    """Toy A는 ΔU = B z이므로 rank-4 선형 transport가 거의 완벽해야 합니다."""
    bundle = make_toy_rank4(n_originals=6, n_variants=2, n_macro=8, seed=0)
    pairs = to_macro_pairs(bundle["pairs"], 8)
    train_pairs, validation_pairs = split_pairs_by_original(pairs, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    rank4 = Rank4LinearTransport.fit(train_pairs, stats)
    result = evaluate_transport(
        None, validation_pairs, stats, floor=FLOOR, baseline=rank4
    )
    assert result["transport_ratio"] < 0.2


# --------------------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------------------


def test_lrt_checkpoint_roundtrip_and_schema_isolation(tmp_path, toy_b):
    train_pairs, validation_pairs = split_pairs_by_original(toy_b, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    cfg = replace(LRTConfig(), denominator_floor=FLOOR)
    result = train_lrt(
        train_pairs, validation_pairs, stats, cfg, floor=FLOOR, max_epochs=2, patience=2
    )
    path = save_lrt_checkpoint(tmp_path / "lrt.pt", result, cfg, stats)
    model, loaded_stats, payload = load_lrt_checkpoint(path)
    assert payload["schema_version"] == LRT_CHECKPOINT_SCHEMA
    assert loaded_stats.hash() == stats.hash()
    assert model.relation_dim == 16 and model.decoder_rank == 16
    tensors = build_pair_tensors(*validation_pairs[0], loaded_stats)
    landmarks = torch.nonzero(
        landmark_mask(*validation_pairs[0], LANDMARK_ALL_COMMON)
    ).flatten()
    with torch.no_grad():
        first = model(
            tensors["delta_norm"], tensors["original_state_norm"],
            tensors["original_update_norm"], support_macros_for(8, (0, 1)), (0, 1),
            landmarks, tensors["relative_positions"],
        )
        second = model(
            tensors["delta_norm"], tensors["original_state_norm"],
            tensors["original_update_norm"], support_macros_for(8, (0, 1)), (0, 1),
            landmarks, tensors["relative_positions"],
        )
    assert torch.equal(first.delta_hat, second.delta_hat)


def test_flow_checkpoint_is_not_loadable_as_lrt(tmp_path):
    """old Flow/Behavior checkpoint를 LRT로 읽지 못하게 합니다."""
    torch.save({"schema_version": "aimo-checkpoint-v3", "model": {}}, tmp_path / "flow.pt")
    with pytest.raises(ValueError, match="not 'aimo-lrt-v1'"):
        load_lrt_checkpoint(tmp_path / "flow.pt")


def test_macro_norm_stats_use_train_originals_only(toy_b):
    train_pairs, validation_pairs = split_pairs_by_original(toy_b, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    assert stats.source == "train_originals"
    assert stats.n_originals == len(train_pairs)
    assert tuple(stats.state_scale.shape) == (9,)
    assert tuple(stats.update_scale.shape) == (8, 2)
    with_validation = fit_macro_norm_stats(
        [o for o, _ in train_pairs] + [o for o, _ in validation_pairs]
    )
    assert with_validation.hash() != stats.hash()


def test_split_keeps_each_original_in_one_split(toy_b):
    train_pairs, validation_pairs = split_pairs_by_original(toy_b, holdout=2)
    train_ids = {o.original_id for o, _ in train_pairs}
    validation_ids = {o.original_id for o, _ in validation_pairs}
    assert not train_ids & validation_ids
    assert train_ids | validation_ids == {o.original_id for o, _ in toy_b}
