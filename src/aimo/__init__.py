"""AIMO: 정답을 유지하는 original-variant pair의 내부 update 흐름 predictor.

Stage 1은 고정 LLM에서 반복 정답을 관측한 pair를 선별하고, 개념/변형 종류 라벨 없이
내부 update 흐름을 자기지도학습합니다. prediction residual은 robustness 지표가
아닙니다.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
