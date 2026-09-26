"""Page 데이터 계약 test."""

from __future__ import annotations

import pytest
import torch

from aimo.page import (
    N_LANDMARKS,
    STREAM_FFN,
    STREAM_MIXER,
    Page,
    common_valid,
    load_pages,
    pair_residual_identity_error,
    save_pages,
    state_diff,
    update_diff,
)


def test_stream_order_and_landmark_count():
    # stream 0 = Mixer, stream 1 = FFN 이고 실제 자료의 P는 16 + 1입니다.
    assert (STREAM_MIXER, STREAM_FFN) == (0, 1)
    assert N_LANDMARKS == 17


def test_shapes_and_residual_identity(datasets):
    page = datasets["train"].groups[0].original
    n_blocks, n_p, hidden = page.n_blocks, page.n_landmarks, page.hidden_size
    assert tuple(page.state.shape) == (n_blocks + 1, n_p, hidden)
    assert tuple(page.updates.shape) == (n_blocks, n_p, 2, hidden)
    assert page.residual_identity_error() == pytest.approx(0.0, abs=1e-5)
    page.validate()


def test_validate_rejects_broken_identity(datasets):
    page = datasets["train"].groups[0].original
    broken = Page(
        state=page.state.clone(),
        updates=page.updates.clone(),
        valid=page.valid.clone(),
        token_offsets=page.token_offsets.clone(),
        relative_positions=page.relative_positions.clone(),
        original_id=page.original_id,
        variant_id=page.variant_id,
    )
    broken.updates[0, 0, 0] += 1.0  # valid landmark의 identity를 깨뜨립니다.
    with pytest.raises(ValueError, match="residual identity"):
        broken.validate()


def test_pair_difference_identity(datasets):
    group = datasets["train"].groups[0]
    variant = next(v for v in group.variants if not v.is_identity)
    d = state_diff(group.original, variant)
    v = update_diff(group.original, variant)
    valid = common_valid(group.original, variant)
    # D[d+1] = D[d] + sum_c V[d, c]
    assert pair_residual_identity_error(d, v, valid) == pytest.approx(0.0, abs=1e-5)


def test_identity_variant_has_zero_update_diff(datasets):
    groups = [g for g in datasets["train"].groups if any(v.is_identity for v in g.variants)]
    assert groups, "identity 대조 example이 있어야 합니다"
    group = groups[0]
    identity = next(v for v in group.variants if v.is_identity)
    assert torch.equal(update_diff(group.original, identity), torch.zeros_like(identity.updates))


def test_save_load_roundtrip(datasets, tmp_path):
    pages = [datasets["train"].groups[0].original]
    save_pages(pages, tmp_path / "pages.npz")
    loaded = load_pages(tmp_path / "pages.npz")
    assert torch.equal(loaded[0].state, pages[0].state)
    assert torch.equal(loaded[0].updates, pages[0].updates)
    assert loaded[0].provenance["source"] == "synthetic"
