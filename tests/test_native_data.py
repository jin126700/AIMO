"""label-free discovery data, locked ID 보호, lineage split, 공식 label null 보존."""

from __future__ import annotations

import json

import pytest

from aimo.native_data import (
    LabelLeak,
    NativePrefix,
    ProtectedRegistry,
    assert_label_free,
    assign_splits,
    generalization_split,
    load_deepmath_discovery,
    load_gsm8k_control,
    load_official_labels,
    make_contrast_probes,
    validate_native_prefix,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_deepmath_discovery_drops_answers_and_solutions(tmp_path):
    rows = [{"id": i, "question": f"Compute {i} + 1.", "final_answer": str(i + 1),
             "r1_solution_1": "long solution", "topic": "algebra", "difficulty": 3.0}
            for i in range(6)]
    problems, report = load_deepmath_discovery(_jsonl(tmp_path / "d.jsonl", rows),
                                               protected=ProtectedRegistry())
    assert report["kept"] == 6
    for problem in problems:
        record = problem.record()
        dumped = json.dumps(record)
        assert "final_answer" not in dumped and "long solution" not in dumped
        assert record["meta"]["topic"] == "algebra"  # split / 분석 metadata로만 남습니다


def test_label_fields_are_rejected_on_the_discovery_path():
    with pytest.raises(LabelLeak):
        assert_label_free({"text": "x", "is_robust": True})
    with pytest.raises(LabelLeak):
        assert_label_free({"text": "x", "meta": {"max_drop": 0.2}})


def test_locked_ids_and_texts_are_protected(tmp_path):
    rows = [{"id": "a", "question": "Locked problem  text."},
            {"id": "b", "question": "Free problem."},
            {"id": "c", "question": "Another locked one."}]
    registry = ProtectedRegistry()
    registry.add_problem("a", None)
    registry.add_problem(None, "another   locked one.".replace("another", "Another"))
    problems, report = load_deepmath_discovery(_jsonl(tmp_path / "d.jsonl", rows),
                                               protected=registry)
    assert [p.problem_id for p in problems] == ["deepmath:b"]
    assert report["protected"] == 2
    cases = _jsonl(tmp_path / "cases.jsonl", [{"id": "case-1", "problem": "Free problem."}])
    assert ProtectedRegistry.load(cases).is_protected(None, "Free  problem.")


def test_gsm_symbolic_and_plus_are_not_mixed_into_discovery(tmp_path):
    path = _jsonl(tmp_path / "g.jsonl", [{"question": "q", "answer": "a"}])
    for source in ("apple/GSM-Symbolic", "qintongli/GSM-Plus"):
        with pytest.raises(ValueError, match="discovery"):
            load_gsm8k_control(path, protected=ProtectedRegistry(), source=source)
    problems, _ = load_gsm8k_control(path, protected=ProtectedRegistry())
    assert "answer" not in problems[0].record()


def test_root_lineage_split_keeps_variants_together(tmp_path):
    rows = [{"id": f"{root}-{k}", "root_id": root, "question": f"Q {root} {k}"}
            for root in range(10) for k in range(3)]
    problems, _ = load_deepmath_discovery(_jsonl(tmp_path / "d.jsonl", rows),
                                          protected=ProtectedRegistry())
    assign_splits(problems, seed=3)
    by_root = {}
    for p in problems:
        by_root.setdefault(p.root_id, set()).add(p.split)
    assert all(len(splits) == 1 for splits in by_root.values())
    held = generalization_split(problems, "topic", {None})
    assert not held["fit"] or not ({p.root_id for p in held["fit"]}
                                   & {p.root_id for p in held["held_out"]})


def test_official_labels_keep_null_and_count_from_source(tmp_path):
    rows = [
        {"model_id": "m1", "dataset_id": "d", "problem_id": "1", "original_problem": "P one.",
         "model_is_robust": True},
        {"model_id": "m2", "dataset_id": "d", "problem_id": "1", "original_problem": "P one.",
         "reasoning_effort": "low", "model_is_robust": None},
        {"model_id": "m1", "dataset_id": "e", "problem_id": "9", "original_problem": "P  one.",
         "model_is_robust": False},
        {"model_id": "m1", "dataset_id": "d", "problem_id": "2", "original_problem": "P two.",
         "model_is_robust": False},
    ]
    labels, summary = load_official_labels(_jsonl(tmp_path / "train.jsonl", rows))
    assert [r.is_robust for r in labels] == [True, None, False, False]
    assert summary["n_labeled"] == 3 and summary["n_null_label"] == 1
    # 같은 text(공백만 다름)는 dataset_id가 달라도 같은 root lineage입니다.
    assert labels[0].root_id == labels[1].root_id == labels[2].root_id != labels[3].root_id
    assert summary["n_unique_roots"] == 2
    assert summary["by_model_effort"]["m2|low"] == {"null": 1}


def test_native_prefix_must_come_from_target_model():
    prefix = NativePrefix("p", "First, note that", "Qwen/Qwen3.5-4B", "h")
    assert validate_native_prefix(prefix, "Qwen/Qwen3.5-4B")
    with pytest.raises(LabelLeak):
        validate_native_prefix(prefix, "other/model")
    with pytest.raises(LabelLeak):
        validate_native_prefix(NativePrefix("p", "x", "Qwen/Qwen3.5-4B", "h", source="dataset"),
                               "Qwen/Qwen3.5-4B")
    with pytest.raises(LabelLeak):
        validate_native_prefix(prefix, "Qwen/Qwen3.5-4B",
                               forbidden_texts=["First, note that x = 2 so"])


def test_contrast_probes_are_fixed_and_labelled_as_probes():
    probes = make_contrast_probes(seed=1)
    assert probes == make_contrast_probes(seed=1)
    assert {p.family for p in probes} == {"number", "sign", "operator", "condition", "style",
                                         "format"}
    assert all("not an answer annotation" in p.note for p in probes)
