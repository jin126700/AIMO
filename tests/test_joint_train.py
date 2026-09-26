"""Joint 학습 test: shared optimizer, checkpoint schema, reload/resume, normalization."""

from __future__ import annotations

import copy

import pytest
import torch

from aimo.config import config_from_dict
from aimo.data import collate_panels, compute_norm_stats, make_synthetic_dataset
from aimo.train import CHECKPOINT_SCHEMA_VERSION, checkpoint_info, load_checkpoint, train
from helpers import tiny_payload


def joint_cfg(tmp_path, **overrides):
    payload = tiny_payload(**overrides)
    payload.setdefault("paths", {})["output_root"] = str(tmp_path)
    payload["model"]["name"] = payload["model"].get("name", "joint")
    payload["train"]["task"] = payload["train"].get("task", "joint")
    return config_from_dict(payload)


def test_joint_training_updates_the_shared_core(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "joint"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 1, "patience": 1},
    )
    summary = train(cfg, datasets)
    model, _, payload = load_checkpoint(tmp_path / "joint" / "last.pt")
    fresh = config_from_dict(tiny_payload(model={"name": "joint"}))
    from aimo.model import build_model

    reference = build_model(
        fresh, datasets["train"].hidden_size, datasets["train"].n_blocks,
        datasets["train"].n_landmarks,
    )
    before = reference.state_dict()
    after = model.state_dict()
    core_changed = [
        key for key in before if key.startswith("flow.core") and not torch.equal(
            before[key], after[key]
        )
    ]
    head_changed = [
        key for key in before if key.startswith("behavior.") and not torch.equal(
            before[key], after[key]
        )
    ]
    flow_head_changed = [
        key for key in before if key.startswith("flow.readout") and not torch.equal(
            before[key], after[key]
        )
    ]
    assert core_changed, "shared core parameter가 학습되어야 합니다"
    assert head_changed, "behavior head가 학습되어야 합니다"
    assert flow_head_changed, "flow readout도 joint에서 함께 학습되어야 합니다"
    assert summary["task"] == "joint"
    assert summary["select_metric"] == "behavior_total"


def test_optimizer_registers_shared_parameters_once(datasets):
    from aimo.model import build_model

    cfg = config_from_dict(tiny_payload(model={"name": "joint"}, train={"task": "joint"}))
    train_set = datasets["train"]
    model = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    params = [p for p in model.parameters() if p.requires_grad]
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids))
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    registered = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(registered) == len(set(registered))


def test_behavior_only_task_does_not_train_the_flow_head(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "beh"}, model={"name": "behavior"},
        train={"task": "behavior", "max_epochs": 1, "patience": 1},
    )
    summary = train(cfg, datasets)
    assert summary["task"] == "behavior"
    model, _, payload = load_checkpoint(tmp_path / "beh" / "last.pt")
    history = summary["history"][-1]
    # behavior task에서는 flow 지표가 정의되지 않습니다.
    assert history["val_flow_total"] != history["val_flow_total"]
    assert history["val_behavior_total"] == history["val_behavior_total"]
    assert model.trained_heads()["pair_drop"] is True


def test_flow_task_stays_available_as_legacy(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "flow"}, model={"name": "loop4"},
        train={"task": "flow", "max_epochs": 1, "patience": 1},
    )
    summary = train(cfg, datasets)
    assert summary["task"] == "flow"
    assert summary["select_metric"] == "total"
    history = summary["history"][-1]
    assert history["val_behavior_total"] != history["val_behavior_total"]  # NaN
    assert history["val_next"] == history["val_next"]


def test_select_metric_must_be_defined_for_the_task():
    """task에서 정의되지 않는 지표를 고르면 시작 전에 막습니다 (best checkpoint 없음 방지)."""
    with pytest.raises(ValueError, match="never defined for"):
        config_from_dict(tiny_payload(train={"task": "flow", "select_metric": "behavior_total"}))
    with pytest.raises(ValueError, match="never defined for"):
        config_from_dict(tiny_payload(train={"task": "behavior", "select_metric": "L_next"}))
    ok = config_from_dict(tiny_payload(train={"task": "flow", "select_metric": "total"}))
    assert ok.train.resolved_select_metric() == "total"


def test_flow_task_saves_a_best_checkpoint(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "flowbest"}, model={"name": "loop4"},
        train={"task": "flow", "select_metric": "total", "max_epochs": 2, "patience": 2},
    )
    summary = train(cfg, datasets)
    assert summary["best_epoch"] >= 0
    assert summary["best_val_total"] is not None
    assert (tmp_path / "flowbest" / "best.pt").exists()


def test_checkpoint_carries_schema_task_and_head_flags(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "schema"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 1, "patience": 1},
    )
    train(cfg, datasets)
    payload = torch.load(tmp_path / "schema" / "last.pt", map_location="cpu")
    info = checkpoint_info(payload)
    assert info["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert info["task"] == "joint"
    assert info["model_name"] == "joint"
    assert info["trained_heads"]["pair_drop"] is True
    assert payload["hashes"]["label_policy_hash"]
    assert payload["hashes"]["data_schema_version"]


def test_v1_style_checkpoint_is_refused(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "old"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 1, "patience": 1},
    )
    train(cfg, datasets)
    path = tmp_path / "old" / "last.pt"
    payload = torch.load(path, map_location="cpu")
    payload.pop("schema_version")  # v1 checkpoint처럼 만듭니다
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema version"):
        load_checkpoint(path)


def test_reload_reproduces_behavior_predictions(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "reload"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 2, "patience": 2},
    )
    train(cfg, datasets)
    model_a, stats_a, _ = load_checkpoint(tmp_path / "reload" / "best.pt")
    model_b, stats_b, _ = load_checkpoint(tmp_path / "reload" / "best.pt")
    batch = collate_panels(datasets["known_test"].groups[:4])
    with torch.no_grad():
        out_a = model_a.forward_behavior(batch.inputs, stats_a)
        panel_a = model_a.panel_outputs(
            out_a, batch.pair_panel, batch.pair_slot, batch.panel_mask
        )
        out_b = model_b.forward_behavior(batch.inputs, stats_b)
        panel_b = model_b.panel_outputs(
            out_b, batch.pair_panel, batch.pair_slot, batch.panel_mask
        )
    assert torch.equal(out_a.pair_drop, out_b.pair_drop)
    assert torch.equal(panel_a.robust_logit, panel_b.robust_logit)
    assert stats_a.hash() == stats_b.hash()


def test_joint_resume_reproduces_uninterrupted_training(tmp_path, datasets, monkeypatch):
    import aimo.train as train_mod

    straight = joint_cfg(
        tmp_path, run={"run_id": "straight"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 3, "patience": 3},
    )
    full = train(straight, datasets)
    crashed = joint_cfg(
        tmp_path, run={"run_id": "resumed"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 3, "patience": 3},
    )
    real = train_mod.validation_metrics
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(train_mod, "validation_metrics", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(crashed, datasets)
    monkeypatch.setattr(train_mod, "validation_metrics", real)
    resumed = train(crashed, datasets, resume=True)
    assert resumed["epochs_run"] == full["epochs_run"] == 3
    for a, b in zip(full["history"], resumed["history"], strict=True):
        assert a["epoch"] == b["epoch"]
        assert a["val_total"] == pytest.approx(b["val_total"], rel=1e-9, abs=1e-12)
        assert a["val_behavior_total"] == pytest.approx(
            b["val_behavior_total"], rel=1e-9, abs=1e-12
        )


def test_resume_refuses_a_different_task(tmp_path, datasets):
    cfg = joint_cfg(
        tmp_path, run={"run_id": "taskguard"}, model={"name": "joint"},
        train={"task": "joint", "max_epochs": 1, "patience": 1},
    )
    train(cfg, datasets)
    # run directory의 config hash가 고정되어 있으므로 task 변경은 config guard에서 막힙니다.
    from aimo.runtime import ConfigMismatchError

    other = joint_cfg(
        tmp_path, run={"run_id": "taskguard"}, model={"name": "behavior"},
        train={"task": "behavior", "max_epochs": 1, "patience": 1},
    )
    with pytest.raises(ConfigMismatchError):
        train(other, datasets, resume=True)


def test_mixed_label_policies_are_blocked(datasets):
    from aimo.train import _check_label_policy

    mixed = copy.deepcopy(datasets)
    group = mixed["train"].groups[0]
    for label in group.pair_labels.values():
        label.policy_hash = "other-policy"
        break
    with pytest.raises(ValueError, match="different model/thinking/sampling policies"):
        _check_label_policy(mixed)


def test_normalization_uses_train_originals_including_failures(datasets):
    """실패 사례를 제외하지 않고 train originals 전체로 scale을 계산합니다."""
    stats_all = compute_norm_stats(datasets["train"])
    non_robust_only = copy.deepcopy(datasets["train"])
    non_robust_only.groups = [
        g for g in non_robust_only.groups
        if g.panel_label is not None and g.panel_label.robust_label == 0
    ]
    assert non_robust_only.groups
    assert compute_norm_stats(non_robust_only).hash() != stats_all.hash()
    assert stats_all.source == "train_originals"


def test_test_split_pages_are_not_used_for_normalization(datasets):
    stats = compute_norm_stats(datasets["train"])
    with_test = copy.deepcopy(datasets["train"])
    with_test.groups = with_test.groups + datasets["known_test"].groups
    assert compute_norm_stats(with_test).hash() != stats.hash()


def test_eval_mode_is_deterministic_and_dropout_off(datasets, stats):
    from aimo.config import config_from_dict as build
    from aimo.model import build_model

    cfg = build(tiny_payload(model={"name": "joint", "dropout": 0.5}, train={"task": "joint"}))
    train_set = datasets["train"]
    model = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    batch = collate_panels(train_set.groups[:3])
    model.eval()
    with torch.no_grad():
        first = model.forward_behavior(batch.inputs, stats).pair_drop
        second = model.forward_behavior(batch.inputs, stats).pair_drop
    assert torch.equal(first, second)
    model.train()
    torch.manual_seed(0)
    a = model.forward_behavior(batch.inputs, stats).pair_drop
    torch.manual_seed(1)
    b = model.forward_behavior(batch.inputs, stats).pair_drop
    assert not torch.equal(a.detach(), b.detach())  # 학습 모드에서는 dropout이 적용됩니다


def test_missing_behavior_supervision_is_reported_not_claimed(tmp_path):
    """behavior supervision이 전혀 없으면 joint 성공으로 보고하지 않습니다."""
    payload = tiny_payload(
        model={"name": "joint"},
        train={"task": "joint", "max_epochs": 1, "patience": 1},
        data={"synthetic": {"behavior_labels": False}},
    )
    payload["paths"] = {"output_root": str(tmp_path)}
    payload["run"] = {"run_id": "nolabel", "seed": 0}
    cfg = config_from_dict(payload)
    datasets = make_synthetic_dataset(cfg)
    summary = train(cfg, datasets)
    assert summary["behavior_supervision_seen"] == {
        "pair_drop": False, "robust": False, "max_drop": False
    }
    assert "warning" in summary
    assert "not a joint training result" in summary["warning"]
