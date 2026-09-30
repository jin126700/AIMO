"""MP formatting view와 edit map endpoint alignment."""

from __future__ import annotations

import re

from aimo.mp_views import (
    VIEW_REFLOW,
    VIEW_WHITESPACE,
    align_endpoints,
    make_mp_views,
    semantic_signature,
)
from aimo.native_data import MP_VERIFIED, classify_variant
from aimo.native_extract import PromptPolicy, render_prompt, tokenize_prompt
from aimo.native_toy import ToyTokenizer

PROBLEM = "Let  x be real.\nIf $3x  +  4 = 10$ and x > -2,\nfind   x."


class CharSpaceTokenizer(ToyTokenizer):
    """whitespace 문자 하나가 token 하나인 변형. view에서 token index가 밀리게 만듭니다."""

    token_pattern = re.compile(r"\s|\d|[A-Za-z]+|[^\sA-Za-z\d]")


def _enc(text: str, tok=None):
    tok = tok or ToyTokenizer()
    return tokenize_prompt(tok, render_prompt(tok, text, PromptPolicy()), 512)


def test_views_preserve_signature_and_math_spans():
    views, skipped = make_mp_views(PROBLEM)
    assert [v.kind for v in views] == [VIEW_WHITESPACE, VIEW_REFLOW]
    for view in views:
        assert semantic_signature(view.text) == semantic_signature(PROBLEM)
        assert "$3x  +  4 = 10$" in view.text  # 수식 span 내부는 그대로
        assert classify_variant(view.record()) == MP_VERIFIED
    assert skipped == {}


def test_no_op_view_is_skipped():
    views, skipped = make_mp_views("Find x if x + 1 = 2.")
    assert views == [] and set(skipped.values()) == {"no_op"}


def test_endpoint_alignment_uses_semantic_prefix_not_token_index():
    view = make_mp_views(PROBLEM)[0][0]
    tok = CharSpaceTokenizer()
    a, b = _enc(PROBLEM, tok), _enc(view.text, tok)
    result = align_endpoints(PROBLEM, view, a["segments"], a["char_spans"], b["segments"],
                             b["char_spans"], a["input_ids"], b["input_ids"])
    assert result.pairs and result.prompt_end is not None
    assert any(i != j for i, j in result.pairs)  # token index가 서로 다릅니다
    for i, j in result.pairs:
        end_a = int(a["char_spans"][i, 1])
        end_b = int(b["char_spans"][j, 1])
        assert semantic_signature(PROBLEM[:end_a]) == semantic_signature(view.text[:end_b])
    assert result.failures.get("whitespace_only_token", 0) > 0
    assert 0 < result.coverage < 1


def test_template_suffix_mismatch_disables_prompt_end_anchor():
    view = make_mp_views(PROBLEM)[0][0]
    a, b = _enc(PROBLEM), _enc(view.text)
    tampered = b["input_ids"].clone()
    tampered[-1] = (tampered[-1] + 1) % 48
    result = align_endpoints(PROBLEM, view, a["segments"], a["char_spans"], b["segments"],
                             b["char_spans"], a["input_ids"], tampered)
    assert result.prompt_end is None
    assert result.failures["template_suffix_mismatch"] == 1


def test_numeric_variant_is_not_mp():
    assert classify_variant({"variant_type": "numeric", "perturbation": "P1"}) != MP_VERIFIED
    assert classify_variant({"mp_view_kind": "paraphrase", "signature_preserved": False}) \
        != MP_VERIFIED
