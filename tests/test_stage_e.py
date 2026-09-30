"""Stage-E encoder / decoder 기하, latent gauge, checkpoint schema 분리."""

from __future__ import annotations

import json

import pytest
import torch

from aimo.native_page import SchemaIncompatible
from aimo.stage_e import (
    CONTROL_STAGE_E,
    STAGE_E_SCHEMA,
    StageE,
    StageECheckpointMeta,
    load_stage_e,
    save_stage_e,
)


def _model(seed=0):
    torch.manual_seed(seed)
    model = StageE(n_macro=3, hidden_size=12, rank=4, vocab_size=20, seed=seed)
    with torch.no_grad():
        model.lower_raw.normal_(0, 0.3)
        model.offset.normal_()
    return model


def test_basis_is_orthonormal_and_lower_invertible():
    model = _model()
    b = model.basis()
    eye = torch.eye(4).expand(3, -1, -1)
    assert torch.allclose(b.transpose(1, 2) @ b, eye, atol=1e-5)
    assert torch.all(torch.linalg.det(model.lower()).abs() > 1e-6)


def test_pseudo_inverse_and_projector():
    model = _model()
    e, e_plus, b = model.encoder_matrix(), model.pseudo_inverse(), model.basis()
    for g in range(3):
        assert torch.allclose(e[g] @ e_plus[g], torch.eye(4), atol=1e-5)
        assert torch.allclose(e_plus[g] @ e[g], b[g] @ b[g].T, atol=1e-5)
        # 일반 E에 대해 E^T E는 projector가 아닙니다.
        ete = e[g].T @ e[g]
        assert not torch.allclose(ete @ ete, ete, atol=1e-3)
    v = torch.randn(5, 12)
    assert torch.allclose(model.project_out(v, 1), v - v @ b[1] @ b[1].T, atol=1e-5)
    captured, total = model.captured_energy(v, 1)
    assert torch.allclose(total - captured, model.project_out(v, 1).pow(2).sum(), atol=1e-4)


def test_same_encoder_entry_exit_identity_and_not_cross_stage():
    model = _model()
    state = torch.randn(2, 4, 7, 12)  # [B, G+1, T, H]
    mixer = torch.randn(2, 3, 7, 12)
    ffn = (state[:, 1:] - state[:, :-1]) - mixer
    z = model.encode_page(state, mixer, ffn)
    assert torch.allclose(z["z_state"] - z["z_in"], z["z_mixer"] + z["z_ffn"], atol=1e-4)
    # 서로 다른 stage encoder의 latent 차이는 update latent와 다릅니다.
    cross = z["z_state"][:, 1] - z["z_state"][:, 0]
    assert not torch.allclose(cross, z["z_mixer"][:, 1] + z["z_ffn"][:, 1], atol=1e-2)


def test_latent_gauge_changes_z_but_not_projector_or_lift():
    model = _model()
    b_before = model.basis().clone()
    x = torch.randn(3, 5, 12)
    z_before = model.encode_update(x.unsqueeze(0))
    with torch.no_grad():
        model.lower_raw.add_(torch.randn_like(model.lower_raw) * 0.5)
    z_after = model.encode_update(x.unsqueeze(0))
    assert not torch.allclose(z_before, z_after, atol=1e-3)
    assert torch.allclose(model.basis(), b_before)
    lifted = torch.stack([model.lift(z_after[0, g], g) for g in range(3)])
    projected = torch.einsum("gth,ghr,gkr->gtk", x, b_before, b_before)
    assert torch.allclose(lifted, projected, atol=1e-4)


def test_decoder_is_linear_in_z_and_shared_across_stages():
    model = _model()
    z = torch.randn(4, 4)
    logits = model.decode_logits(z)
    assert logits.shape == (4, 20)
    assert torch.allclose(model.decode_logits(2 * z) - model.decode_logits(z),
                          model.decode_logits(z) - model.decoder_bias, atol=1e-5)
    assert model.decoder.shape == (20, 4)  # stage 축이 없습니다 (공유 decoder)


def _meta():
    return StageECheckpointMeta(
        control=CONTROL_STAGE_E, rank=4, n_macro=3, hidden_size=12, vocab_size=20,
        sensitivity_lambda=0.1, npr_topk=8, sketch={"q": 8, "seed": 1},
        native={"model_id": "toy", "model_revision": "rev"}, dataset={"split": "toy"},
    )


def test_checkpoint_roundtrip_and_schema(tmp_path):
    model = _model()
    save_stage_e(model, _meta(), tmp_path)
    loaded, meta = load_stage_e(tmp_path)
    assert meta["schema"] == STAGE_E_SCHEMA
    assert torch.allclose(loaded.encoder_matrix(), model.encoder_matrix(), atol=1e-6)
    assert not any(p.requires_grad for p in loaded.parameters())  # frozen encoder


def test_legacy_checkpoints_are_not_loaded_as_stage_e(tmp_path, cfg, datasets, stats):
    from aimo.lrt_experiment import LRT_CHECKPOINT_SCHEMA
    from aimo.train import CHECKPOINT_SCHEMA_VERSION, load_checkpoint

    with pytest.raises(SchemaIncompatible):
        load_stage_e(tmp_path)  # 빈 directory / legacy checkpoint
    (tmp_path / "stage_e.json").write_text(json.dumps({"schema": LRT_CHECKPOINT_SCHEMA}))
    with pytest.raises(SchemaIncompatible):
        load_stage_e(tmp_path)
    (tmp_path / "stage_e.json").write_text(json.dumps({"schema": CHECKPOINT_SCHEMA_VERSION}))
    with pytest.raises(SchemaIncompatible):
        load_stage_e(tmp_path)
    # 반대로 legacy loader는 Stage-E checkpoint를 읽지 않습니다.
    good = tmp_path / "good"
    save_stage_e(_model(), _meta(), good)
    with pytest.raises(Exception):  # noqa: B017 - legacy loader가 어떤 식으로든 거부해야 합니다.
        load_checkpoint(good / "stage_e.pt")


def test_tampered_checkpoint_is_rejected(tmp_path):
    save_stage_e(_model(), _meta(), tmp_path)
    blob = torch.load(tmp_path / "stage_e.pt", weights_only=True)
    blob["offset"] += 1
    torch.save(blob, tmp_path / "stage_e.pt")
    with pytest.raises(SchemaIncompatible, match="sha256"):
        load_stage_e(tmp_path)
