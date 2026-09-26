"""Scorer regression test: nested boxed, exact 비교, thinking 경계 사례."""

from __future__ import annotations

from aimo.adapters.qwen import (
    SCORER_VERSION,
    SPLIT_CLOSED,
    SPLIT_EMPTY,
    SPLIT_NO_THINKING,
    SPLIT_UNCLOSED,
    SUBMIT_CONFLICTING,
    SUBMIT_NONE,
    SUBMIT_SINGLE,
    classify_thinking_slot,
    parse_submitted_answer,
    score_answer,
    split_thinking,
)
from aimo.scoring import (
    SUPPORTED_FORMS,
    VERDICT_CORRECT,
    VERDICT_UNSUPPORTED,
    VERDICT_WRONG,
    compare_answers,
    extract_boxed,
    parse_answer,
)


def slot(**kwargs) -> str:
    base = {"started": True, "infra_error": False, "hit_cap": False, "is_final_cap": False}
    base.update(kwargs)
    return classify_thinking_slot(**base)


# --------------------------------------------------------------------------------------
# 필수 regression cases
# --------------------------------------------------------------------------------------


def test_half_and_zero_point_five_are_equivalent():
    assert compare_answers("1/2", "0.5") == VERDICT_CORRECT
    assert compare_answers(r"\frac{1}{2}", "0.5") == VERDICT_CORRECT
    assert compare_answers(r"\dfrac{2}{4}", "0.50") == VERDICT_CORRECT


def test_large_integers_are_not_collapsed_by_float():
    """float 비교는 이 둘을 같다고 판정합니다. exact 비교는 구분해야 합니다."""
    assert float("9007199254740993") == float("9007199254740992")  # float의 한계
    assert compare_answers("9007199254740993", "9007199254740992") == VERDICT_WRONG
    assert compare_answers("9007199254740993", "9007199254740993") == VERDICT_CORRECT


def test_answer_only_inside_thinking_is_not_correct():
    text = "<think>the answer is 42</think> I could not finish."
    assert slot(text=text, gold="42") == "U_score"
    split = split_thinking(text)
    assert split.state == SPLIT_CLOSED
    assert "42" in split.thinking and "42" not in split.answer_region


def test_nested_boxed_fraction_is_extracted():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} holds") == [r"\frac{1}{2}"]
    assert slot(text=r"<think>x</think> \boxed{\frac{1}{2}}", gold="0.5") == "C"
    assert extract_boxed(r"\boxed{\frac{1}") == []  # 닫히지 않은 boxed는 버립니다


def test_unsupported_expression_is_u_score_not_wrong():
    assert compare_answers(r"\sqrt{2}", "1.4142") == VERDICT_UNSUPPORTED
    assert slot(text=r"<think>x</think> \boxed{\sqrt{2}}", gold="1.4142") == "U_score"
    assert parse_answer(r"\sqrt{2}").kind not in SUPPORTED_FORMS


def test_conflicting_final_answers_follow_the_documented_rule():
    """상충하는 답은 채점 불가, 동치인 여러 답은 채택합니다."""
    conflicting = parse_submitted_answer(r"\boxed{7} and \boxed{8}")
    assert conflicting.status == SUBMIT_CONFLICTING and conflicting.raw is None
    assert slot(text=r"<think>x</think> \boxed{7} and \boxed{8}", gold="7") == "U_score"
    equivalent = parse_submitted_answer(r"\boxed{1/2} then \boxed{0.5}")
    assert equivalent.status == SUBMIT_SINGLE
    assert slot(text=r"<think>x</think> \boxed{1/2} then \boxed{0.5}", gold="0.5") == "C"


def test_scorer_version_was_raised():
    assert SCORER_VERSION == "2"


# --------------------------------------------------------------------------------------
# split_thinking 경계 사례
# --------------------------------------------------------------------------------------


def test_prompt_opened_thinking_with_only_a_closing_tag():
    """chat template이 <think>를 열어 둔 경우 generated suffix에는 </think>만 있습니다."""
    text = "reasoning 42</think> Final answer: 7"
    split = split_thinking(text, thinking_already_open=True)
    assert split.state == SPLIT_CLOSED
    assert split.answer_region.strip() == "Final answer: 7"
    assert "42" in split.thinking
    assert slot(text=text, gold="7", thinking_already_open=True) == "C"
    # flag를 주지 않아도 </think>가 먼저 나오면 열려 있던 것으로 추론합니다.
    inferred = split_thinking(text)
    assert inferred.state == SPLIT_CLOSED
    assert inferred.answer_region.strip() == "Final answer: 7"
    assert slot(text=text, gold="7") == "C"
    # thinking 안의 42는 답으로 쓰이지 않습니다.
    assert slot(text=text, gold="42") == "W"


def test_both_tags_in_generated_text():
    split = split_thinking("<think>maybe 42</think> answer is 7")
    assert split.state == SPLIT_CLOSED
    assert split.answer_region.strip() == "answer is 7"


def test_unclosed_thinking_has_no_answer_region():
    split = split_thinking("<think>still working, maybe 42")
    assert split.state == SPLIT_UNCLOSED
    assert split.answer_region == ""
    assert slot(text="<think>still working, 42", gold="42", hit_cap=True) == "X"


def test_plain_non_thinking_answer():
    split = split_thinking(r"The answer is \boxed{7}")
    assert split.state == SPLIT_NO_THINKING
    assert slot(text=r"The answer is \boxed{7}", gold="7") == "C"


def test_only_special_tokens_generated_then_truncated():
    special = frozenset({151643, 151667})
    split = split_thinking(
        "", generated_token_ids=[151643, 151667], special_token_ids=special
    )
    assert split.state == SPLIT_EMPTY
    assert slot(
        text="",
        gold="7",
        generated_token_ids=[151643],
        special_token_ids=special,
    ) == "X"
    # 의미 있는 token이 있으면 empty가 아닙니다.
    assert split_thinking(
        "<think>x</think> 7", generated_token_ids=[151643, 5], special_token_ids=special
    ).state == SPLIT_CLOSED


def test_multiple_thinking_blocks_use_the_same_rule():
    split = split_thinking(r"<think>a</think>mid<think>b</think> \boxed{7}")
    assert split.state == SPLIT_CLOSED
    assert "a" in split.thinking and "b" in split.thinking
    assert slot(text=r"<think>a</think>mid<think>b</think> \boxed{7}", gold="7") == "C"


def test_no_submitted_answer_distinguishes_cap_hit_from_unscored():
    assert parse_submitted_answer("nothing here").status == SUBMIT_NONE
    assert slot(text="<think>x</think> hmm", gold="7", hit_cap=False) == "U_score"
    assert slot(text="<think>x</think> hmm", gold="7", hit_cap=True) == "X"


def test_normalization_rules_are_bounded():
    """정규화는 표면 표기만 다루고 값을 바꾸지 않습니다."""
    assert compare_answers("$7$", "7") == VERDICT_CORRECT
    assert compare_answers("1,234", "1234") == VERDICT_CORRECT
    assert compare_answers("+7.0", "7") == VERDICT_CORRECT
    assert compare_answers("-3", "3") == VERDICT_WRONG
    assert compare_answers("1/0", "1") == VERDICT_UNSUPPORTED  # 0으로 나누기는 미지원
    assert score_answer(None, "7") == VERDICT_UNSUPPORTED
