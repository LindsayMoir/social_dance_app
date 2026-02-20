import sys

sys.path.insert(0, 'src')

from llm import LLMHandler


def test_trim_extracted_text_for_budget_returns_original_when_within_budget():
    text = "abc" * 100
    out = LLMHandler._trim_extracted_text_for_budget(text, 1000)
    assert out == text


def test_trim_extracted_text_for_budget_truncates_and_keeps_head_tail():
    text = ("HEAD-" * 3000) + ("TAIL-" * 3000)
    out = LLMHandler._trim_extracted_text_for_budget(text, 5000)
    assert len(out) <= 5000 + 50  # allow delimiter overhead
    assert out.startswith("HEAD-")
    assert out.endswith("TAIL-")
    assert "[TRUNCATED]" in out


def test_trim_extracted_text_for_budget_handles_empty_and_nonpositive_budget():
    assert LLMHandler._trim_extracted_text_for_budget("", 5000) == ""
    assert LLMHandler._trim_extracted_text_for_budget("hello", 0) == ""
