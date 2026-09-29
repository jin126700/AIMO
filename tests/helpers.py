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


def fine_page(
    n_blocks: int = 32,
    n_landmarks: int = 4,
    hidden: int = 8,
    seed: int = 0,
    *,
    variant: bool = False,
    invalid: tuple[int, ...] = (),
):
    """residual identity를 정확히 만족하는 Fine Page fixture (MacroPage/LRT test 공용)."""
    import torch

    from aimo.page import Page

    generator = torch.Generator().manual_seed(seed)
    updates = torch.randn(n_blocks, n_landmarks, 2, hidden, generator=generator) * 0.3
    state = torch.zeros(n_blocks + 1, n_landmarks, hidden)
    state[0] = torch.randn(n_landmarks, hidden, generator=generator)
    for depth in range(n_blocks):
        state[depth + 1] = state[depth] + updates[depth].sum(dim=1)
    valid = torch.ones(n_landmarks, dtype=torch.bool)
    for index in invalid:
        valid[index] = False
    return Page(
        state=state,
        updates=updates,
        valid=valid,
        token_offsets=torch.arange(n_landmarks, dtype=torch.long),
        relative_positions=torch.linspace(0.0, 1.0, n_landmarks),
        original_id="o",
        variant_id="o#v1" if variant else "o#orig",
        provenance={"source": "synthetic"},
    )
