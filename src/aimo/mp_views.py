"""보수적 MP(meaning-preserving) formatting view와 edit map alignment.

view는 식·숫자·부호·변수·조건을 보존해야 합니다. 이를 보장하려고 **수식 span 밖의
whitespace만** 바꾸는 view 두 가지만 씁니다. 공백이 아닌 문자의 순서(semantic signature)가
원문과 정확히 같지 않으면 view를 버립니다. test-time LLM paraphrase나 긴 reasoning
generation은 기본 경로에 넣지 않습니다.

token index끼리 차이를 계산하지 않습니다. edit map과 tokenizer offset으로 **같은 semantic
prefix에서 끝나는 complete-span endpoint**를 대응시킵니다. 원문 token이 끝나는 문자 위치를
view의 문자 위치로 옮겼을 때, 그 위치에서 정확히 끝나는 view token이 있을 때만 대응합니다.
whitespace만으로 된 token 등 대응이 없는 경우는 실패 이유와 함께 기록합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .native_page import SEG_PROBLEM

VIEW_WHITESPACE = "whitespace_collapse"
VIEW_REFLOW = "line_reflow"
VIEW_KINDS = (VIEW_WHITESPACE, VIEW_REFLOW)
MAX_VIEWS = 2

# 수식 span. 이 안의 문자는 절대 바꾸지 않습니다.
_MATH_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)


def semantic_signature(text: str) -> str:
    """공백이 아닌 문자의 순서. 숫자·부호·연산자·변수·조건은 모두 여기에 들어 있습니다."""
    return re.sub(r"\s+", "", text)


def _outside_math(text: str, transform) -> str:
    out, cursor = [], 0
    for match in _MATH_RE.finditer(text):
        out.append(transform(text[cursor : match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(transform(text[cursor:]))
    return "".join(out)


def _collapse(segment: str) -> str:
    segment = re.sub(r"[ \t]+", " ", segment)
    return re.sub(r" *\n *", "\n", segment)


def _reflow(segment: str) -> str:
    segment = re.sub(r"\n{3,}", "\n\n", segment)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", segment)


@dataclass
class MPView:
    kind: str
    text: str
    signature_preserved: bool
    char_map: list[int]  # 원문 비공백 문자 위치 -> view 문자 위치 (-1: 공백)

    def record(self) -> dict:
        return {"mp_view_kind": self.kind, "signature_preserved": self.signature_preserved}


def build_char_map(original: str, view: str) -> list[int]:
    """비공백 문자끼리 순서대로 대응시킵니다. 공백 문자는 -1입니다."""
    mapping = [-1] * len(original)
    j = 0
    for i, char in enumerate(original):
        if char.isspace():
            continue
        while j < len(view) and view[j].isspace():
            j += 1
        if j >= len(view) or view[j] != char:
            raise ValueError("view does not preserve the non-whitespace character sequence")
        mapping[i] = j
        j += 1
    return mapping


def make_mp_views(problem: str, kinds: tuple[str, ...] = VIEW_KINDS) -> tuple[list[MPView], dict]:
    """최대 2개의 view. 원문과 같거나 signature가 바뀐 view는 버리고 이유를 남깁니다."""
    transforms = {VIEW_WHITESPACE: _collapse, VIEW_REFLOW: _reflow}
    views, skipped = [], {}
    for kind in kinds[:MAX_VIEWS]:
        text = _outside_math(problem, transforms[kind])
        if text == problem:
            skipped[kind] = "no_op"
            continue
        if semantic_signature(text) != semantic_signature(problem):
            skipped[kind] = "signature_changed"
            continue
        views.append(MPView(kind, text, True, build_char_map(problem, text)))
    return views, skipped


@dataclass
class Alignment:
    """원문 token index와 view token index의 endpoint 대응."""

    pairs: list[tuple[int, int]] = field(default_factory=list)
    n_candidates: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    prompt_end: tuple[int, int] | None = None

    @property
    def coverage(self) -> float:
        return len(self.pairs) / self.n_candidates if self.n_candidates else 0.0

    def fail(self, reason: str) -> None:
        self.failures[reason] = self.failures.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {
            "n_pairs": len(self.pairs),
            "n_candidates": self.n_candidates,
            "coverage": self.coverage,
            "failures": dict(self.failures),
            "prompt_end_aligned": self.prompt_end is not None,
        }

    def index_tensors(self) -> tuple[Tensor, Tensor]:
        if not self.pairs:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        orig, view = zip(*self.pairs, strict=True)
        return torch.tensor(orig), torch.tensor(view)


def align_endpoints(
    original_problem: str,
    view: MPView,
    orig_segments: Tensor,
    orig_spans: Tensor,
    view_segments: Tensor,
    view_spans: Tensor,
    orig_ids: Tensor | None = None,
    view_ids: Tensor | None = None,
) -> Alignment:
    """문제 본문 token의 complete-span endpoint를 대응시킵니다.

    span은 `native_extract.tokenize_prompt`가 만든 문제 text 기준 char offset입니다. 두
    prompt의 마지막 token(generation 시작 직전)은 문제 뒤 template suffix token이 두 prompt에서
    같을 때만 prompt-end anchor로 대응시킵니다.
    """
    result = Alignment()
    view_end_to_token: dict[int, int] = {}
    for index in range(view_segments.shape[0]):
        if int(view_segments[index]) == SEG_PROBLEM:
            view_end_to_token[int(view_spans[index, 1])] = index
    for index in range(orig_segments.shape[0]):
        if int(orig_segments[index]) != SEG_PROBLEM:
            continue
        result.n_candidates += 1
        start, end = int(orig_spans[index, 0]), int(orig_spans[index, 1])
        last = next((p for p in range(end - 1, start - 1, -1)
                     if not original_problem[p].isspace()), None)
        if last is None:
            result.fail("whitespace_only_token")
            continue
        view_end = view.char_map[last] + 1
        match = view_end_to_token.get(view_end)
        if match is None:
            result.fail("no_view_token_ends_at_endpoint")
            continue
        result.pairs.append((index, match))
    if orig_ids is not None and view_ids is not None:
        if not torch.equal(_suffix(orig_ids, orig_segments), _suffix(view_ids, view_segments)):
            result.fail("template_suffix_mismatch")
            return result
    result.prompt_end = (int(orig_segments.shape[0]) - 1, int(view_segments.shape[0]) - 1)
    return result


def _suffix(ids: Tensor, segments: Tensor) -> Tensor:
    """마지막 문제 token 뒤의 token id들."""
    problem = (segments == SEG_PROBLEM).nonzero()
    last = int(problem.max()) if problem.numel() else -1
    return ids[last + 1 :]
