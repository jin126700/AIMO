"""Test 공용 config payload. 작고 빠른 자료(H=8, L=4, P=3)를 씁니다."""

from __future__ import annotations

BASE = {
    "run": {"run_id": "test", "seed": 0},
    "data": {
        "synthetic": {
            "hidden_size": 8,
            "n_blocks": 4,
            "n_landmarks": 3,
            "n_originals_train": 6,
            "n_originals_validation": 3,
            "n_originals_test": 3,
            "variants_per_original": 3,
        }
    },
    "model": {"name": "loop4", "d_model": 16, "n_heads": 2, "ffn_dim": 32, "dropout": 0.0},
    "train": {
        "max_epochs": 2,
        "patience": 2,
        "batch_originals": 3,
        "microbatch_originals": 3,
        "cuts_per_pair": 1,
    },
    "eval": {"bootstrap_samples": 0},
}


def tiny_payload(**overrides) -> dict:
    """BASE를 얕은 병합으로 덮어씁니다."""
    import copy

    payload = copy.deepcopy(BASE)
    for section, values in overrides.items():
        node = payload.setdefault(section, {})
        for key, value in values.items():
            if isinstance(value, dict):
                node.setdefault(key, {}).update(value)
            else:
                node[key] = value
    return payload
