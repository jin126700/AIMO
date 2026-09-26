"""서버 전용 optional adapter들.

Core CPU test suite는 이 package를 import하지 않고도 동작해야 합니다. 여기의
기능은 transformers나 MathGAP 같은 optional 의존성을 요구하며, 확인하지 못한 외부
API는 SERVER_PENDING으로 표시합니다.
"""

from __future__ import annotations


class AdapterUnavailable(RuntimeError):
    """optional 의존성이나 설정이 없어 adapter를 쓸 수 없습니다 (fail-fast)."""


SERVER_PENDING = "SERVER_PENDING"

__all__ = ["AdapterUnavailable", "SERVER_PENDING"]
