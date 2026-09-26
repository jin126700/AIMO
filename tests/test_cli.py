"""CLI test. 기본 실행은 dry-run이며 model weights를 로드하지 않습니다."""

from __future__ import annotations

import json

import pytest

from aimo.cli import main


def write_config(tmp_path, extra: str = "") -> str:
    path = tmp_path / "toy.yaml"
    path.write_text(
        "run:\n"
        "  run_id: cli\n"
        "  seed: 0\n"
        "  device: cpu\n"
        "paths:\n"
        f"  output_root: {tmp_path / 'runs'}\n"
        "data:\n"
        "  source: synthetic\n"
        "  synthetic:\n"
        "    hidden_size: 8\n"
        "    n_blocks: 4\n"
        "    n_landmarks: 3\n"
        "    n_originals_train: 6\n"
        "    n_originals_validation: 3\n"
        "    n_originals_test: 3\n"
        "model:\n"
        "  name: loop4\n"
        "  d_model: 16\n"
        "  n_heads: 2\n"
        "  ffn_dim: 32\n"
        "  dropout: 0.0\n"
        "train:\n"
        "  max_epochs: 2\n"
        "  patience: 2\n"
        "  batch_originals: 3\n"
        "  microbatch_originals: 3\n"
        "  cuts_per_pair: 1\n"
        "eval:\n"
        "  bootstrap_samples: 0\n" + extra,
        encoding="utf-8",
    )
    return str(path)


def run(capsys, *argv) -> dict:
    code = main(list(argv))
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out)


def test_check_reports_the_page_contract(tmp_path, capsys):
    payload = run(capsys, "check", "--config", write_config(tmp_path))
    assert payload["status"] == "ok"
    assert payload["page_contract"]["streams"] == ["mixer", "ffn"]
    assert payload["model"]["params"]["core"] > 0
    assert payload["adapters"]["mathgap"]["status"] == "SERVER_PENDING"


def test_make_toy_writes_pages(tmp_path, capsys):
    config = write_config(tmp_path)
    payload = run(capsys, "make-toy", "--config", config, "--out", str(tmp_path / "pages"))
    assert set(payload["splits"]) == {
        "train",
        "validation",
        "known_test",
        "unseen_perturbation_test",
        "harder_test",
    }
    assert (tmp_path / "pages" / "index.json").exists()
    # 저장한 Page를 다시 읽어 학습할 수 있습니다.
    reloaded = run(
        capsys,
        "check",
        "--config",
        config,
        "--set",
        "data.source=pages",
        "--set",
        f"data.page_dir={tmp_path / 'pages'}",
    )
    assert reloaded["status"] == "ok"


def test_train_evaluate_predict_roundtrip(tmp_path, capsys):
    config = write_config(tmp_path)
    trained = run(capsys, "train", "--config", config)
    assert trained["epochs_run"] >= 1
    evaluated = run(capsys, "evaluate", "--config", config, "--splits", "known_test")
    metrics = evaluated["splits"]["known_test"]["metrics"]
    assert metrics["next_mse_norm"]["n_originals"] == 3
    assert "support_swap_degradation" in metrics
    predicted = run(capsys, "predict", "--config", config, "--split", "known_test", "--cut", "1")
    assert predicted["cut"] == 1
    assert predicted["v_hat_shape"] == [3, 2, 8]


def test_evaluate_compare_collects_runs(tmp_path, capsys):
    config = write_config(tmp_path)
    for name in ("loop4", "persistence"):
        run(capsys, "train", "--config", config, "--set", f"model.name={name}", "--run-id", name)
        run(
            capsys,
            "evaluate",
            "--config",
            config,
            "--set",
            f"model.name={name}",
            "--run-id",
            name,
            "--splits",
            "known_test",
        )
    payload = run(
        capsys,
        "evaluate",
        "--config",
        config,
        "--compare",
        str(tmp_path / "runs" / "loop4"),
        str(tmp_path / "runs" / "persistence"),
    )
    assert {row["model"] for row in payload["rows"]} == {"loop4", "persistence"}


def test_preflight_does_not_load_weights(tmp_path, capsys):
    payload = run(capsys, "preflight", "--config", write_config(tmp_path))
    assert payload["dry_run"] is True
    assert payload["loaded_model_weights"] is False
    assert payload["gpu_budget"]["can_start"] is True
    assert payload["screening_plan"]["slots_per_prompt"] == 4
    assert any("Qwen3-4B" in item for item in payload["server_pending"])


def test_extract_tiny_runs_on_cpu(tmp_path, capsys):
    pytest.importorskip("transformers", reason="optional server dependency")
    payload = run(
        capsys,
        "extract",
        "--config",
        write_config(tmp_path),
        "--tiny",
        "--hidden-size",
        "16",
        "--layers",
        "2",
        "--prompt-len",
        "24",
    )
    assert payload["real_weights"] is False
    assert payload["skipped"] is False
    assert payload["residual_identity_max_error"] < 1e-4
    assert payload["state_shape"] == [3, 17, 16]
    assert payload["n_extracted_total"] == 1


def test_extract_dedup_skips_a_repeated_prompt(tmp_path, capsys):
    pytest.importorskip("transformers", reason="optional server dependency")
    config = write_config(tmp_path)
    argv = ["extract", "--config", config, "--tiny", "--hidden-size", "16", "--layers", "2"]
    first = run(capsys, *argv)
    assert first["skipped"] is False
    second = run(capsys, *argv)  # 같은 prompt는 ledger로 건너뜁니다.
    assert second["skipped"] is True
    assert "dedup ledger" in second["reason"]


def test_gpu_budget_blocks_a_real_extract(tmp_path, capsys):
    from aimo.runtime import GpuBudget

    config = write_config(tmp_path)
    run_dir = tmp_path / "runs" / "cli"
    run_dir.mkdir(parents=True, exist_ok=True)
    GpuBudget.open(run_dir).add(200 * 60)  # 누적 200분: 새 GPU 작업을 막습니다.
    assert main(["extract", "--config", config, "--execute-gpu"]) == 2
    assert "budget exhausted" in capsys.readouterr().err


def test_unsupported_stages_fail_fast(tmp_path, capsys):
    config = write_config(tmp_path)
    assert main(["screen", "--config", config]) == 2
    assert main(["extract", "--config", config]) == 2  # --tiny 없이 실제 weights 요구
    assert main(["prepare-data", "--config", config]) == 2
    err = capsys.readouterr().err
    assert "SERVER_PENDING" in err or "--execute-gpu" in err


def test_run_stage1_stops_at_the_first_unsupported_stage(tmp_path, capsys):
    config = write_config(tmp_path)
    code = main(["run-stage1", "--config", config])
    out = json.loads(capsys.readouterr().out)
    assert code == 2
    assert out["status"] == "stopped"
    assert out["executed"] == []
    assert out["stopped_at"]["stage"] == "prepare-data"


def test_run_stage1_can_run_the_supported_stages(tmp_path, capsys):
    config = write_config(tmp_path)
    code = main(["run-stage1", "--config", config, "--stages", "train"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["executed"] == ["train"]
    assert (tmp_path / "runs" / "cli" / "best.pt").exists()


def test_missing_page_directory_fails_with_a_clear_message(tmp_path, capsys):
    config = write_config(tmp_path)
    assert (
        main(
            [
                "check",
                "--config",
                config,
                "--set",
                "data.source=pages",
                "--set",
                f"data.page_dir={tmp_path / 'nope'}",
            ]
        )
        == 2
    )
    assert "missing index.json" in capsys.readouterr().err


def test_gpu_device_requires_the_explicit_flag(tmp_path, capsys):
    config = write_config(tmp_path, extra="")
    assert main(["check", "--config", config, "--set", "run.device=cuda"]) == 2
    assert "--execute-gpu" in capsys.readouterr().err
