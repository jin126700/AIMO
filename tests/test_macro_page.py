"""MacroPage-8 derived view test. Fine Page 계약은 바뀌지 않아야 합니다."""

from __future__ import annotations

import pytest
import torch

from aimo.macro_page import (
    DEFAULT_N_MACRO,
    MACRO_SCHEMA,
    common_valid,
    macro_boundaries,
    macro_relation,
    path_energy,
    relation_energy,
    to_macro_page,
)
from helpers import fine_page


@pytest.mark.parametrize("n_blocks", [24, 32, 36, 48])
def test_uneven_layer_counts_map_to_eight_macro_stages(n_blocks):
    """24 / 32 / 36 / 48 layer 모두 같은 G=8 coordinate를 씁니다 (4 layer hard-code 없음)."""
    bounds = macro_boundaries(n_blocks, DEFAULT_N_MACRO)
    assert len(bounds) == DEFAULT_N_MACRO + 1
    assert bounds[0] == 0 and bounds[-1] == n_blocks
    assert all(right > left for left, right in zip(bounds, bounds[1:], strict=False))
    macro = to_macro_page(fine_page(n_blocks=n_blocks), DEFAULT_N_MACRO)
    assert macro.n_macro == DEFAULT_N_MACRO
    assert tuple(macro.state.shape) == (DEFAULT_N_MACRO + 1, 4, 8)
    assert tuple(macro.updates.shape) == (DEFAULT_N_MACRO, 4, 2, 8)


def test_macro_requires_g_not_greater_than_l():
    with pytest.raises(ValueError, match="must be <= n_blocks"):
        macro_boundaries(4, 8)
    with pytest.raises(ValueError, match="must be <= n_blocks"):
        to_macro_page(fine_page(n_blocks=4), 8)


@pytest.mark.parametrize("n_blocks", [24, 32, 36])
def test_macro_update_is_the_exact_fine_sum(n_blocks):
    page = fine_page(n_blocks=n_blocks)
    macro = to_macro_page(page, DEFAULT_N_MACRO)
    bounds = macro.boundaries
    for g in range(DEFAULT_N_MACRO):
        expected = page.updates[bounds[g] : bounds[g + 1]].sum(dim=0)
        assert torch.equal(macro.updates[g], expected)
    for g in range(DEFAULT_N_MACRO + 1):
        assert torch.equal(macro.state[g], page.state[bounds[g]])


@pytest.mark.parametrize("n_blocks", [24, 32, 36, 48])
def test_macro_residual_identity_holds(n_blocks):
    macro = to_macro_page(fine_page(n_blocks=n_blocks), DEFAULT_N_MACRO)
    assert macro.residual_identity_error() < 1e-3
    macro.validate()
    lhs = macro.state[1:]
    rhs = macro.state[:-1] + macro.updates.sum(dim=2)
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_source_fine_page_is_not_mutated():
    page = fine_page()
    before_state = page.state.clone()
    before_updates = page.updates.clone()
    before_fingerprint = page.content_fingerprint()
    macro = to_macro_page(page, DEFAULT_N_MACRO, with_path_energy=True)
    macro.state.add_(1.0)
    macro.updates.add_(1.0)
    assert torch.equal(page.state, before_state)
    assert torch.equal(page.updates, before_updates)
    assert page.content_fingerprint() == before_fingerprint


def test_macro_fingerprint_is_stable_and_schema_aware():
    page = fine_page()
    first = to_macro_page(page, 8).fingerprint()
    second = to_macro_page(page, 8).fingerprint()
    assert first == second  # same Fine Page + same G -> same fingerprint
    assert to_macro_page(page, 4).fingerprint() != first  # G가 다르면 달라집니다
    assert to_macro_page(fine_page(seed=1), 8).fingerprint() != first
    provenance = to_macro_page(page, 8).provenance
    assert provenance["macro_schema"] == MACRO_SCHEMA
    assert provenance["source_page_fingerprint"] == page.content_fingerprint()
    assert provenance["source_provenance"]["source"] == "synthetic"


def test_path_energy_is_exact_and_diagnostic_only():
    page = fine_page()
    macro = to_macro_page(page, 8, with_path_energy=True)
    bounds = macro.boundaries
    energy = path_energy(page, list(bounds))
    assert tuple(energy.shape) == (8, 4, 2)
    for g in range(8):
        expected = page.updates[bounds[g] : bounds[g + 1]].norm(dim=-1).sum(dim=0)
        assert torch.allclose(energy[g], expected, atol=1e-6)
    assert torch.allclose(macro.path_energy, energy)
    # 기본 생성에서는 만들지 않습니다 (학습 경로에 들어가지 않는 diagnostic).
    assert to_macro_page(page, 8).path_energy is None


def test_valid_and_token_metadata_are_preserved():
    page = fine_page(invalid=(1,))
    macro = to_macro_page(page, 8)
    assert torch.equal(macro.valid, page.valid)
    assert torch.equal(macro.token_offsets, page.token_offsets)
    assert torch.equal(macro.relative_positions, page.relative_positions)
    assert macro.original_id == page.original_id
    assert macro.variant_id == page.variant_id
    assert macro.n_blocks == page.n_blocks


def test_relation_uses_common_validity_only():
    original = to_macro_page(fine_page(invalid=(1,)), 8)
    variant = to_macro_page(fine_page(seed=1, variant=True, invalid=(2,)), 8)
    mask = common_valid(original, variant)
    assert mask.tolist() == [True, False, False, True]
    delta = macro_relation(original, variant)
    assert torch.equal(delta, variant.updates - original.updates)
    # relation energy는 공통 valid landmark만 셉니다.
    energy = relation_energy(delta, mask)
    manual = float((delta[:, mask].pow(2)).sum())
    assert energy == pytest.approx(manual, rel=1e-6)


def test_relation_requires_the_same_original():
    original = to_macro_page(fine_page(), 8)
    other = to_macro_page(fine_page(seed=2), 8)
    object.__setattr__(other, "original_id", "different")
    with pytest.raises(ValueError, match="share original_id"):
        macro_relation(original, other)


def test_existing_page_save_load_is_unchanged(tmp_path):
    """LRT 때문에 기존 Page artifact를 다시 추출할 필요가 없어야 합니다."""
    from aimo.page import load_pages, save_pages

    page = fine_page()
    save_pages([page], tmp_path / "pages.npz")
    loaded = load_pages(tmp_path / "pages.npz")[0]
    assert torch.equal(loaded.state, page.state)
    assert torch.equal(loaded.updates, page.updates)
    assert loaded.content_fingerprint() == page.content_fingerprint()
    # 같은 Fine Page에서 같은 MacroPage가 나옵니다.
    assert to_macro_page(loaded, 8).fingerprint() == to_macro_page(page, 8).fingerprint()
