"""Looped architecture와 leakage 차단 test.

shared parameter 재사용, untied 독립성, variant future 불변성, reference를 통한 간접
leakage 차단을 확인합니다.
"""

from __future__ import annotations

import copy

import pytest
import torch

from aimo.config import config_from_dict
from aimo.data import collate, sample_pairs
from aimo.model import (
    KIND_OBS,
    KIND_QUERY,
    KIND_REF,
    build_attention_mask,
    build_layout,
    build_model,
)
from helpers import tiny_payload


def make(name: str, datasets):
    cfg = config_from_dict(tiny_payload(model={"name": name}))
    train = datasets["train"]
    return build_model(cfg, train.hidden_size, train.n_blocks, train.n_landmarks)


def test_shared_block_is_one_object_called_n_loops_times(datasets, stats, batch):
    model = make("loop4", datasets)
    assert len(model.blocks) == 1  # 독립 copy 4개가 아니라 같은 객체 하나입니다.
    calls = []
    handle = model.blocks[0].register_forward_hook(lambda *_: calls.append(1))
    model.eval()
    model(batch.input_a, stats)
    handle.remove()
    assert len(calls) == model.n_loops == 4


def test_loop1_and_loop4_have_identical_parameter_counts(datasets):
    loop1 = make("loop1", datasets).param_report()
    loop4 = make("loop4", datasets).param_report()
    assert (loop1.total, loop1.core, loop1.input_output) == (
        loop4.total,
        loop4.core,
        loop4.input_output,
    )


def test_untied_blocks_are_independent(datasets):
    model = make("untied4", datasets)
    assert len(model.blocks) == 4
    ids = [id(p) for block in model.blocks for p in block.parameters()]
    assert len(ids) == len(set(ids))  # parameter 객체가 공유되지 않습니다.
    core_untied = model.param_report().core
    core_tied = make("loop4", datasets).param_report().core
    assert core_untied == 4 * core_tied
    before = [b.ffn[0].weight.detach().clone() for b in model.blocks]
    with torch.no_grad():
        model.blocks[0].ffn[0].weight.zero_()
    assert torch.equal(model.blocks[1].ffn[0].weight, before[1])
    assert torch.equal(model.blocks[3].ffn[0].weight, before[3])


def test_reference_rows_never_read_observed_or_query_cells(datasets):
    layout = build_layout(n_blocks=4, n_landmarks=3, cut=2, use_variant_prefix=True,
                          device=torch.device("cpu"))
    valid = torch.ones(1, 3, dtype=torch.bool)
    blocked = build_attention_mask(layout, valid[:, layout.landmark])[0]
    allowed = ~blocked
    ref_rows = layout.kind == KIND_REF
    non_ref_cols = layout.kind != KIND_REF
    assert not bool(allowed[ref_rows][:, non_ref_cols].any())
    # observed depth r는 reference와 observed depth <= r만 읽습니다.
    obs_rows = torch.nonzero(layout.kind == KIND_OBS).flatten()
    for i in obs_rows.tolist():
        cols = torch.nonzero(allowed[i]).flatten()
        kinds = layout.kind[cols]
        assert not bool((kinds == KIND_QUERY).any())
        later = (kinds == KIND_OBS) & (layout.depth[cols] > layout.depth[i])
        assert not bool(later.any())
    # query는 reference와 prefix <= cut만 읽습니다.
    query_rows = torch.nonzero(layout.kind == KIND_QUERY).flatten()
    for i in query_rows.tolist():
        cols = torch.nonzero(allowed[i]).flatten()
        kinds = layout.kind[cols]
        obs_depths = layout.depth[cols][kinds == KIND_OBS]
        assert bool((obs_depths <= layout.cut).all())


def test_padding_landmarks_do_not_create_all_masked_rows(datasets):
    layout = build_layout(4, 3, 2, True, torch.device("cpu"))
    valid = torch.tensor([[True, False, False]])
    blocked = build_attention_mask(layout, valid[:, layout.landmark])
    assert bool((~blocked).any(dim=-1).all()), "모든 row에 최소 하나의 key가 열려 있어야 합니다"


def _corrupt_future(datasets, cut: int):
    """cut 이후의 variant state/updates를 크게 바꾼 dataset 복제본을 만듭니다."""
    corrupted = copy.deepcopy(datasets)
    for group in corrupted["train"].groups:
        for variant in group.variants:
            variant.state[cut + 1 :] += 7.0
            variant.updates[cut:] -= 5.0
    return corrupted


@pytest.mark.parametrize("name", ["loop4", "m0", "linear"])
def test_prediction_is_invariant_to_variant_future(datasets, stats, name):
    cut = 1
    model = make(name, datasets)
    model.eval()

    def predict(source):
        generator = torch.Generator().manual_seed(0)
        samples = sample_pairs(source["train"], generator, 1)
        for sample in samples:
            sample.cut = cut
        with torch.no_grad():
            return model(collate(samples).input_a, stats)

    clean = predict(datasets)
    dirty = predict(_corrupt_future(datasets, cut))
    assert torch.equal(clean, dirty)


def test_reference_activations_do_not_change_with_variant_prefix(datasets, stats):
    """reference cell 표현은 observed variant prefix에 의존하지 않습니다."""
    model = make("loop4", datasets)
    model.eval()
    generator = torch.Generator().manual_seed(0)
    samples = sample_pairs(datasets["train"], generator, 1)
    for sample in samples:
        sample.cut = 2
    batch = collate(samples)
    layout = build_layout(
        datasets["train"].n_blocks, datasets["train"].n_landmarks, 2, True, torch.device("cpu")
    )
    ref_slice = torch.nonzero(layout.kind == KIND_REF).flatten()

    captured: list[torch.Tensor] = []
    handle = model.blocks[0].register_forward_hook(
        lambda _m, _i, out: captured.append(out.detach().clone())
    )
    with torch.no_grad():
        model(batch.input_a, stats)  # variant a의 prefix
        first = captured[-1][:, ref_slice].clone()
        captured.clear()
        model(batch.input_b, stats)  # 다른 variant의 prefix
        second = captured[-1][:, ref_slice].clone()
    handle.remove()
    assert torch.equal(first, second)


def test_m0_ignores_the_entire_variant_prefix(datasets, stats, batch):
    model = make("m0", datasets)
    model.eval()
    with torch.no_grad():
        base = model(batch.input_a, stats)
        swapped = copy.deepcopy(batch.input_a)
        swapped.var_state_prefix = torch.randn_like(swapped.var_state_prefix)
        swapped.var_updates_prefix = torch.randn_like(swapped.var_updates_prefix)
        other = model(swapped, stats)
    assert torch.equal(base, other)


def test_query_cells_carry_no_activation(datasets, stats, batch):
    """query cell feature는 전부 0이어야 합니다 (target activation 미포함)."""
    model = make("loop4", datasets)
    layout = build_layout(
        batch.input_a.n_blocks, datasets["train"].n_landmarks, batch.cut, True,
        torch.device("cpu"),
    )
    features = model._cell_features(batch.input_a, stats, layout)
    query = features[:, layout.query_index]
    assert torch.count_nonzero(query) == 0


def test_persistence_predicts_zero(datasets, stats, batch):
    model = make("persistence", datasets)
    out = model(batch.input_a, stats)
    assert torch.count_nonzero(out) == 0
    assert model.param_report().total == 0
