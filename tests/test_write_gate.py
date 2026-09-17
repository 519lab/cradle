"""Write-quality gate (enhancement #3): a bad response must not be cached."""

from __future__ import annotations

import pytest

from cradle.gateway.writeback import cache_skip_reason


def _completion(finish: str | None, content: str | None) -> dict:
    return {
        "choices": [
            {"index": 0, "finish_reason": finish,
             "message": {"role": "assistant", "content": content}}
        ]
    }


def test_stop_with_content_is_cacheable() -> None:
    assert cache_skip_reason(_completion("stop", "Paris")) is None


def test_eos_is_cacheable() -> None:
    assert cache_skip_reason(_completion("eos", "Paris")) is None


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", None])
def test_bad_finish_reason_skipped(finish: str | None) -> None:
    reason = cache_skip_reason(_completion(finish, "partial answer"))
    assert reason == f"finish_{finish}"


def test_empty_content_skipped() -> None:
    assert cache_skip_reason(_completion("stop", "")) == "empty_content"


def test_whitespace_content_skipped() -> None:
    assert cache_skip_reason(_completion("stop", "   \n")) == "empty_content"


def test_none_content_skipped() -> None:
    assert cache_skip_reason(_completion("stop", None)) == "empty_content"


def test_no_choices_skipped() -> None:
    assert cache_skip_reason({"choices": []}) == "no_choices"
    assert cache_skip_reason({}) == "no_choices"
