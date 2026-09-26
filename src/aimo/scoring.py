"""Final answer 추출과 exact 비교.

float 비교는 큰 정수에서 서로 다른 값을 같다고 판정하므로 쓰지 않습니다. 대신 지원 범위를
명시하고 그 안에서만 `fractions.Fraction`으로 정확히 비교합니다. 범위를 벗어난 표현은
**오답이 아니라 채점 불가(unsupported)** 로 보고합니다.

지원 범위 (SUPPORTED_FORMS):

- integer: 임의 정밀도 정수 (`-12`, `+7`, `1,234` -> 1234)
- decimal: 유한 소수 (`0.5`, `-2.50`) — Decimal을 거쳐 정확히 비교합니다
- fraction: `a/b`, `\\frac{a}{b}`, `\\dfrac{a}{b}`, `\\tfrac{a}{b}` (a, b는 정수)

지원하지 않는 것: 기호(pi, e, x), 근호, 지수, 구간, 순서쌍, 집합, 단위, 백분율, 그 밖의
임의 수식. **임의 eval이나 untrusted expression 실행을 하지 않습니다.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

SUPPORTED_FORMS = ("integer", "decimal", "fraction")
KIND_UNSUPPORTED = "unsupported"
KIND_EMPTY = "empty"

# 비교 결과. wrong과 unsupported를 구분합니다.
VERDICT_CORRECT = "correct"
VERDICT_WRONG = "wrong"
VERDICT_UNSUPPORTED = "unsupported"

_LATEX_FRACTION = re.compile(
    r"^\\(?:d|t)?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}$"
)
_PLAIN_FRACTION = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_INTEGER = re.compile(r"^[+-]?\d+$")
_DECIMAL = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)$")
# 세 자리 묶음 천 단위 구분자만 제거합니다 (좌표 (1,2) 같은 것을 뭉개지 않습니다).
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


@dataclass(frozen=True)
class AnswerValue:
    """정규화한 답과 그 해석 결과."""

    raw: str
    normalized: str
    kind: str  # SUPPORTED_FORMS 중 하나 또는 unsupported / empty
    value: Fraction | None = None

    @property
    def supported(self) -> bool:
        return self.kind in SUPPORTED_FORMS


def normalize_answer_text(value: str) -> str:
    """비교 전 표면 정규화. 값 자체를 바꾸는 변환은 하지 않습니다."""
    text = value.strip()
    # 수식 구분자와 흔한 wrapper 제거
    for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        while text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left) : -len(right)].strip()
    text = text.rstrip(".").strip()
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = _THOUSANDS.sub("", text)
    text = re.sub(r"\s+", "", text)
    return text


def parse_answer(value: str | None) -> AnswerValue:
    """지원 범위 안에서만 Fraction으로 해석합니다. 그 밖에는 unsupported입니다."""
    if value is None:
        return AnswerValue(raw="", normalized="", kind=KIND_EMPTY)
    normalized = normalize_answer_text(value)
    if not normalized:
        return AnswerValue(raw=value, normalized="", kind=KIND_EMPTY)

    latex = _LATEX_FRACTION.match(normalized)
    if latex:
        numerator, denominator = int(latex.group(1)), int(latex.group(2))
        if denominator == 0:
            return AnswerValue(value, normalized, KIND_UNSUPPORTED)
        return AnswerValue(value, normalized, "fraction", Fraction(numerator, denominator))

    plain = _PLAIN_FRACTION.match(normalized)
    if plain:
        numerator, denominator = int(plain.group(1)), int(plain.group(2))
        if denominator == 0:
            return AnswerValue(value, normalized, KIND_UNSUPPORTED)
        return AnswerValue(value, normalized, "fraction", Fraction(numerator, denominator))

    if _INTEGER.match(normalized):
        # 임의 정밀도 정수. float로 내리지 않습니다.
        return AnswerValue(value, normalized, "integer", Fraction(int(normalized), 1))

    if _DECIMAL.match(normalized):
        try:
            return AnswerValue(value, normalized, "decimal", Fraction(Decimal(normalized)))
        except (InvalidOperation, ValueError):
            return AnswerValue(value, normalized, KIND_UNSUPPORTED)

    return AnswerValue(value, normalized, KIND_UNSUPPORTED)


def compare_answers(predicted: str | None, gold: str) -> str:
    """predicted와 gold를 비교해 correct / wrong / unsupported를 돌려줍니다.

    - 양쪽이 모두 지원 범위이면 Fraction 정확 비교입니다 (1/2 == 0.5).
    - 한쪽이라도 지원 범위를 벗어나면, 정규화 문자열이 정확히 같을 때만 correct이고
      그렇지 않으면 **wrong이 아니라 unsupported**입니다 (채점 불가).
    - predicted가 비어 있으면 unsupported입니다 (제출된 답이 없음).
    """
    left = parse_answer(predicted)
    right = parse_answer(gold)
    if left.kind == KIND_EMPTY:
        return VERDICT_UNSUPPORTED
    if left.supported and right.supported:
        return VERDICT_CORRECT if left.value == right.value else VERDICT_WRONG
    if left.normalized and left.normalized == right.normalized:
        return VERDICT_CORRECT
    return VERDICT_UNSUPPORTED


def extract_boxed(text: str) -> list[str]:
    r"""`\boxed{...}`의 내용을 brace matching으로 뽑습니다.

    중첩 brace를 지원합니다: `\boxed{\frac{1}{2}}` -> `\frac{1}{2}`.
    닫히지 않은 `\boxed{`는 무시합니다.
    """
    results: list[str] = []
    marker = "\\boxed"
    index = 0
    while True:
        found = text.find(marker, index)
        if found == -1:
            return results
        cursor = found + len(marker)
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            index = found + len(marker)
            continue
        depth = 0
        start = cursor + 1
        while cursor < len(text):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start:cursor])
                    break
            cursor += 1
        if depth != 0:  # 닫히지 않은 boxed는 버립니다.
            return results
        index = cursor + 1
