"""MathGAP 최소 adapter와 legacy screening grading.

v2에서 MathGAP/GSM 경로는 **구현·저난도 대조용 legacy**입니다. primary 데이터 경로는
`adapters/deepmath.py`이고, 어려운 문제용 실행 protocol은 `adapters/qwen.py`의 research
thinking profile입니다. 이 module의 non-thinking parser를 새 경로에서 재사용하지 않습니다.

MathGAP의 공식 API를 추측하지 않습니다. AIMO가 필요한 것은 세 개의 callable이며,
서버 운영자가 확인한 실제 API 경로를 config의 dotted path로 지정합니다.

    generator(spec: dict) -> list[dict]   # original/variant 문제 spec 목록
    renderer(problem: dict) -> str        # prompt 문자열
    oracle(problem: dict) -> str          # 정답 문자열

경로가 비어 있으면 SERVER_PENDING으로 fail-fast합니다. 빈 성공 파일이나 허위 완료
결과를 만들지 않습니다.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from . import SERVER_PENDING, AdapterUnavailable

# screening slot outcome 코드. cap-hit은 X이며 W로 합치지 않습니다.
OUTCOME_CORRECT = "C"
OUTCOME_WRONG = "W"
OUTCOME_CAP_HIT = "X"
OUTCOME_UNSCORED = "U_score"
OUTCOME_INFRA_ERROR = "infra_error"
OUTCOME_NOT_STARTED = "not_started"
OUTCOMES = (
    OUTCOME_CORRECT,
    OUTCOME_WRONG,
    OUTCOME_CAP_HIT,
    OUTCOME_UNSCORED,
    OUTCOME_INFRA_ERROR,
    OUTCOME_NOT_STARTED,
)

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_LABEL = re.compile(r"(?:final answer|answer)\s*[:=]\s*(-?[\d.,/]+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def probe_mathgap() -> dict:
    """mathgap package 설치 여부만 보고합니다. API 형태를 단정하지 않습니다."""
    try:
        module = importlib.import_module("mathgap")
    except ImportError:
        return {"available": False, "status": SERVER_PENDING, "version": None}
    return {
        "available": True,
        "status": SERVER_PENDING,
        "version": getattr(module, "__version__", "unknown"),
        "note": "설치는 확인했지만 generator/renderer/oracle 경로는 config로 지정해야 합니다.",
    }


def _resolve(path: str | None, role: str) -> Callable:
    if not path:
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: MathGAP {role} path is not configured "
            f"(set server.mathgap.{role}_path to a verified dotted path)"
        )
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise AdapterUnavailable(f"{role}_path must be a dotted path, got {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: cannot import {module_name!r} for MathGAP {role}: {exc}"
        ) from exc
    target = getattr(module, attr, None)
    if not callable(target):
        raise AdapterUnavailable(f"{module_name}.{attr} is not callable")
    return target


@dataclass
class MathGapAdapter:
    """확인된 dotted path 세 개를 묶은 얇은 adapter."""

    generator: Callable
    renderer: Callable
    oracle: Callable
    revision: str

    @classmethod
    def from_config(cls, mathgap_cfg) -> MathGapAdapter:
        return cls(
            generator=_resolve(mathgap_cfg.generator_path, "generator"),
            renderer=_resolve(mathgap_cfg.renderer_path, "renderer"),
            oracle=_resolve(mathgap_cfg.oracle_path, "oracle"),
            revision=mathgap_cfg.revision,
        )


# --------------------------------------------------------------------------------------
# 고정 final-answer parser와 exact oracle
# --------------------------------------------------------------------------------------


def parse_final_answer(text: str) -> str | None:
    """고정 규칙 parser. 연구용 screening이며 공식 AIMO 평가 정책이 아닙니다.

    우선순위: 마지막 \\boxed{...} -> 마지막 "answer:" 라벨 -> 마지막 숫자.
    """
    boxed = _BOXED.findall(text)
    if boxed:
        return _clean(boxed[-1])
    labelled = _ANSWER_LABEL.findall(text)
    if labelled:
        return _clean(labelled[-1])
    numbers = _NUMBER.findall(text)
    if numbers:
        return _clean(numbers[-1])
    return None


def _clean(value: str) -> str:
    return value.strip().rstrip(".").replace(",", "").replace(" ", "")


def exact_match(predicted: str | None, gold: str) -> bool:
    """exact oracle. 숫자는 정규화 후 비교하고, 그 외는 문자열 일치를 씁니다."""
    if predicted is None:
        return False
    left, right = _clean(predicted), _clean(gold)
    try:
        return abs(float(left) - float(right)) < 1e-9
    except ValueError:
        return left == right


def classify_slot(
    *,
    started: bool,
    infra_error: bool,
    hit_cap: bool,
    text: str | None,
    gold: str,
) -> str:
    """slot 하나의 outcome. C/W/X/U_score/infra_error/not_started를 분리합니다."""
    if not started:
        return OUTCOME_NOT_STARTED
    if infra_error:
        return OUTCOME_INFRA_ERROR
    predicted = parse_final_answer(text or "")
    if predicted is None:
        # cap에 걸려 답을 못 낸 경우는 X이며 W로 합치지 않습니다.
        return OUTCOME_CAP_HIT if hit_cap else OUTCOME_UNSCORED
    if hit_cap:
        return OUTCOME_CAP_HIT
    return OUTCOME_CORRECT if exact_match(predicted, gold) else OUTCOME_WRONG


@dataclass
class PromptGrade:
    """unique prompt 하나에 대한 독립 4slot 결과."""

    outcomes: list[str]

    @property
    def counts(self) -> dict[str, int]:
        return {code: self.outcomes.count(code) for code in OUTCOMES}

    @property
    def n_slots(self) -> int:
        return len(self.outcomes)

    @property
    def is_c4(self) -> bool:
        """4/4 성공. population robustness 인증이 아닙니다."""
        return self.n_slots == 4 and all(code == OUTCOME_CORRECT for code in self.outcomes)


def pair_eligible(
    semantic_valid: bool, original: PromptGrade, variant: PromptGrade
) -> tuple[bool, str]:
    """Eligibility = semantic-valid AND original C4 AND variant C4."""
    if not semantic_valid:
        return False, "semantic_invalid"
    if not original.is_c4:
        return False, "original_not_c4"
    if not variant.is_c4:
        return False, "variant_not_c4"
    return True, "eligible"
