"""Stage-E discovery pipeline (CPU toy), Dev 선택 규칙, config, CLI."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aimo.cli import main
from aimo.config import Config, StageEConfig, config_from_dict
from aimo.stage_e import STAGE_E_SCHEMA, load_stage_e
from aimo.stage_e_experiment import (
    CONTROL_FULL_HIDDEN,
    CONTROL_LOCAL_ORACLE,
    CONTROL_NPR_ONLY,
    CONTROL_PCA,
    CONTROL_RANDOM,
    CONTROL_REDUCED_RANK,
    CONTROL_STAGE_E,
    STATUS_TOY,
    losses,
    run_toy_experiment,
    select_on_dev,
)

TOY_CFG = StageEConfig(n_macro=4, max_tokens=512, lambda_grid=(0.0, 1.0))


@pytest.fixture(scope="module")
def toy_report(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("stage_e_toy")
    return run_dir, run_toy_experiment(TOY_CFG, run_dir, epochs=15)


def test_toy_pipeline_reports_all_controls_and_evaluations(toy_report):
    run_dir, report = toy_report
    assert report["status"] == STATUS_TOY
    held = report["A_held_out_fidelity"]
    for control in (CONTROL_STAGE_E, CONTROL_NPR_ONLY, CONTROL_RANDOM, CONTROL_PCA,
                    CONTROL_REDUCED_RANK, CONTROL_FULL_HIDDEN, CONTROL_LOCAL_ORACLE):
        assert control in held
    for key in ("coarse_topk_other_kl", "full_vocab_kl_audit", "sketch_relative_error",
                "independent_sketch_relative_error", "gradient_capture"):
        assert key in held[CONTROL_STAGE_E]
    assert held[CONTROL_FULL_HIDDEN]["gradient_capture"] == pytest.approx(1.0, abs=1e-5)
    assert held[CONTROL_NPR_ONLY]["lambda"] == 0.0
    assert {"mp_family=format_views"} <= set(report["B_generalization"])
    assert set(report["continuation_audit_full_kl"]) >= {"number", "sign", "operator"}
    controls = report["C_intervention"][0]["recovery"]
    assert set(controls) == {"projected", "random_subspace", "complement", "norm_matched_random",
                             "rank_matched_pca", "site_matched_random_donor"}
    model, meta = load_stage_e(run_dir / "stage_e")
    assert meta["schema"] == STAGE_E_SCHEMA and STATUS_TOY in meta["notes"]
    assert meta["rank"] in TOY_CFG.rank_grid and meta["selection"]["rank"] == meta["rank"]
    assert json.loads((run_dir / "stage_e_report.json").read_text())["status"] == STATUS_TOY


def test_selection_only_sees_dev_and_follows_declared_rule():
    def row(rank, lam, kl, cap, split="dev"):
        return {"control": CONTROL_STAGE_E, "rank": rank, "lambda": lam, "split": split,
                "metrics": {"coarse_topk_other_kl": kl, "gradient_capture": cap}}

    rows = [row(4, 0.0, 0.30, 0.2), row(8, 0.0, 0.100, 0.3), row(8, 1.0, 0.104, 0.9),
            row(16, 0.0, 0.098, 0.4), row(16, 1.0, 0.200, 0.95)]
    chosen = select_on_dev(rows, tolerance=0.05)
    assert (chosen["rank"], chosen["lambda"]) == (8, 1.0)
    with pytest.raises(ValueError, match="Dev"):
        select_on_dev(rows + [row(4, 0.0, 0.01, 1.0, split="held_out")], tolerance=0.05)


def test_positive_lambda_requires_vjp_batches(toy_report):
    from aimo.native_extract import PromptPolicy
    from aimo.native_toy import ToyTokenizer, build_toy_hybrid
    from aimo.stage_e import StageE
    from aimo.stage_e_experiment import NativeTeacher, toy_problems

    teacher = NativeTeacher(build_toy_hybrid(), ToyTokenizer(), TOY_CFG,
                            PromptPolicy(max_tokens=512), topk=8)
    batch = teacher.batch([toy_problems()[0].record()], need_vjp=False)
    model = StageE(4, 16, 4, 48)
    assert "sensitivity" not in losses(model, batch, 0.0, TOY_CFG)
    with pytest.raises(ValueError, match="VJP"):
        losses(model, batch, 1.0, TOY_CFG)


def test_stage_e_config_validation_and_legacy_hash_preserved():
    with pytest.raises(ValueError):
        StageEConfig(trust_remote_code=True)
    with pytest.raises(ValueError):
        StageEConfig(lambda_grid=(0.1, 1.0))  # lambda=0 baseline이 빠짐
    with pytest.raises(ValueError):
        StageEConfig(audit_sketch_seed=StageEConfig().sketch_seed)
    with pytest.raises(ValueError):
        config_from_dict({"stage_e": {"unknown_key": 1}})
    base = Config()
    assert base.hash() == config_from_dict({}).hash()
    changed = replace(base, stage_e=replace(base.stage_e, rank=8))
    assert changed.hash() != base.hash()  # 실제로 쓰는 stage_e 설정은 hash에 들어갑니다


def test_cli_spec_toy_extract_and_blocked_real_paths(tmp_path, capsys):
    assert main(["stage-e", "--run-dir", str(tmp_path), "--stage", "spec"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["backend"] == "stage_e_v1" and spec["unverified"]
    assert main(["native-extract", "--run-dir", str(tmp_path / "x"), "--toy", "--n-macro", "4",
                 "--limit", "3"]) == 0
    assert (tmp_path / "x" / "native_pages" / "native_index.json").exists()
    capsys.readouterr()
    # pin되지 않은 revision이나 데이터 없이 실제 단계를 부르면 막힙니다.
    assert main(["stage-e", "--run-dir", str(tmp_path), "--stage", "discovery"]) == 2
    assert main(["official-contract"]) == 0
    assert "de794053" in capsys.readouterr().out


def test_real_discovery_reports_data_unavailable(tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "stage_e:\n  model_revision: abc\n  tokenizer_revision: abc\n"
        f"  protected_registry: {tmp_path / 'missing.json'}\n", encoding="utf-8")
    code = main(["stage-e", "--run-dir", str(tmp_path / "r"), "--stage", "discovery",
                 "--config", str(cfg_path)])
    assert code == 3
    assert "DATA_UNAVAILABLE" in capsys.readouterr().out
