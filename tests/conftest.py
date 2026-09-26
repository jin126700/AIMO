"""공용 fixture. test는 작고 빠른 synthetic Page만 씁니다 (H=8, L=4, P=3)."""

from __future__ import annotations

import pytest
import torch

from aimo.config import Config, config_from_dict
from aimo.data import (
    collate,
    compute_norm_stats,
    group_by_cut,
    make_synthetic_dataset,
    sample_pairs,
)
from helpers import tiny_payload


@pytest.fixture
def cfg() -> Config:
    return config_from_dict(tiny_payload())


@pytest.fixture
def datasets(cfg: Config):
    return make_synthetic_dataset(cfg)


@pytest.fixture
def stats(datasets):
    return compute_norm_stats(datasets["train"])


@pytest.fixture
def batch(datasets):
    generator = torch.Generator().manual_seed(0)
    samples = sample_pairs(datasets["train"], generator, 1)
    for sample in samples:
        sample.cut = 1  # 고정 cut으로 결정적인 test를 만듭니다.
    return collate(samples)


@pytest.fixture
def bucketed(datasets):
    generator = torch.Generator().manual_seed(1)
    return group_by_cut(sample_pairs(datasets["train"], generator, 1))
