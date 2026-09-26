"""실제 gradient/parameter update와 checkpoint reload/resume test."""

from __future__ import annotations

import copy

import pytest
import torch

from aimo.config import config_from_dict
from aimo.data import collate, compute_norm_stats, make_synthetic_dataset, sample_pairs
from aimo.losses import compute_loss
from aimo.model import build_model
from aimo.train import load_checkpoint, train
from helpers import tiny_payload


def cfg_for(tmp_path, **overrides):
    payload = tiny_payload(**overrides)
    payload.setdefault("paths", {})["output_root"] = str(tmp_path)
    return config_from_dict(payload)


def test_backward_produces_real_gradients(datasets, stats, batch):
    cfg = config_from_dict(tiny_payload())
    train_set = datasets["train"]
    model = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    model.train()
    compute_loss(model, batch, stats).total.backward()
    core = model.blocks[0]
    grads = [p.grad for p in core.parameters() if p.grad is not None]
    assert grads, "shared block에 gradient가 흘러야 합니다"
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert model.readout.weight.grad is not None
    assert float(model.readout.weight.grad.abs().sum()) > 0


def test_training_updates_parameters(tmp_path, datasets):
    cfg = cfg_for(tmp_path, run={"run_id": "upd"}, train={"max_epochs": 1, "patience": 1})
    train_set = datasets["train"]
    reference = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    before = copy.deepcopy(reference.state_dict())
    summary = train(cfg, datasets)
    model, _, _ = load_checkpoint(tmp_path / "upd" / "last.pt")
    after = model.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "학습 후 parameter가 바뀌어야 합니다"
    assert summary["epochs_run"] == 1
    assert summary["params"]["core"] > 0


def test_checkpoint_reload_reproduces_predictions(tmp_path, datasets):
    cfg = cfg_for(tmp_path, run={"run_id": "reload"}, train={"max_epochs": 2, "patience": 2})
    train(cfg, datasets)
    model_a, stats_a, payload = load_checkpoint(tmp_path / "reload" / "best.pt")
    model_b, stats_b, _ = load_checkpoint(tmp_path / "reload" / "best.pt")
    generator = torch.Generator().manual_seed(7)
    samples = sample_pairs(datasets["known_test"], generator, 1)
    for sample in samples:
        sample.cut = 1
    batch = collate(samples)
    with torch.no_grad():
        first = model_a(batch.input_a, stats_a)
        second = model_b(batch.input_a, stats_b)
    assert torch.equal(first, second)
    assert stats_a.hash() == stats_b.hash() == payload["hashes"]["stats_hash"]


def test_checkpoint_stores_hashes_and_optimizer_state(tmp_path, datasets):
    cfg = cfg_for(tmp_path, run={"run_id": "hashes"}, train={"max_epochs": 1, "patience": 1})
    train(cfg, datasets)
    payload = torch.load(tmp_path / "hashes" / "last.pt", map_location="cpu")
    hashes = payload["hashes"]
    assert hashes["config_hash"] == cfg.hash()
    assert hashes["stats_hash"] == compute_norm_stats(datasets["train"]).hash()
    assert set(hashes["split_hashes"]) == set(datasets)
    assert payload["optimizer"] is not None
    assert payload["torch_rng_state"] is not None
    assert payload["sampler_rng_state"] is not None
    assert payload["model_meta"]["n_landmarks"] == datasets["train"].n_landmarks


def test_resume_reproduces_uninterrupted_training(tmp_path, datasets, monkeypatch):
    """중간에 죽은 run을 같은 config로 resume하면 끊기지 않은 학습과 같은 값이 나옵니다."""
    import aimo.train as train_mod

    straight = cfg_for(tmp_path, run={"run_id": "straight"}, train={"max_epochs": 3, "patience": 3})
    full = train(straight, datasets)

    crashed = cfg_for(tmp_path, run={"run_id": "resumed"}, train={"max_epochs": 3, "patience": 3})
    real_validation = train_mod.validation_loss
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # epoch 2의 validation에서 죽은 상황을 만듭니다.
            raise RuntimeError("simulated crash")
        return real_validation(*args, **kwargs)

    monkeypatch.setattr(train_mod, "validation_loss", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(crashed, datasets)
    monkeypatch.setattr(train_mod, "validation_loss", real_validation)

    payload = torch.load(tmp_path / "resumed" / "last.pt", map_location="cpu")
    assert payload["epoch"] == 1  # epoch 0, 1까지만 저장되어 있습니다.

    resumed = train(crashed, datasets, resume=True)
    assert resumed["epochs_run"] == full["epochs_run"] == 3
    for a, b in zip(full["history"], resumed["history"], strict=True):
        assert a["epoch"] == b["epoch"]
        assert a["val_total"] == pytest.approx(b["val_total"], rel=1e-9, abs=1e-12)
        assert a["train_total"] == pytest.approx(b["train_total"], rel=1e-9, abs=1e-12)
    assert resumed["best_epoch"] == full["best_epoch"]


def test_resume_refuses_a_changed_config(tmp_path, datasets):
    from aimo.runtime import ConfigMismatchError

    cfg = cfg_for(tmp_path, run={"run_id": "cfgguard"}, train={"max_epochs": 1, "patience": 1})
    train(cfg, datasets)
    changed = cfg_for(tmp_path, run={"run_id": "cfgguard"}, train={"max_epochs": 5, "patience": 1})
    with pytest.raises(ConfigMismatchError):
        train(changed, datasets, resume=True)


def test_resume_refuses_a_different_dataset(tmp_path, datasets):
    cfg = cfg_for(tmp_path, run={"run_id": "guard"}, train={"max_epochs": 1, "patience": 1})
    train(cfg, datasets)
    other = make_synthetic_dataset(
        config_from_dict(
            tiny_payload(run={"seed": 99}, data={"synthetic": {"n_originals_train": 5}})
        )
    )
    with pytest.raises(ValueError, match="mismatch|do not match"):
        train(cfg, other, resume=True)
