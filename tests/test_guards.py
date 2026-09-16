from __future__ import annotations

from cradle.compress.guards import extract_protected, restore_protected
from cradle.compress.rules import strip_fluff


def test_fence_roundtrip() -> None:
    text = "intro\n```python\nprint(1)\n```\noutro"
    ph, masked, spans = extract_protected(text)
    assert any(s.reason == "code" for s in spans)
    assert "print(1)" not in masked
    assert restore_protected(masked, ph) == text


def test_unclosed_fence_to_eof() -> None:
    text = "```json\n{\"a\":1}"
    ph, masked, spans = extract_protected(text)
    assert spans and spans[0].reason == "code"
    assert restore_protected(masked, ph) == text


def test_almost_json_not_protected() -> None:
    text = "Use {name} in the template please"
    _ph, _masked, spans = extract_protected(text)
    assert not any(s.reason == "json" for s in spans)


def test_format_line() -> None:
    text = "Do the thing\nrespond only in JSON\n"
    _ph, _masked, spans = extract_protected(text)
    assert any(s.reason == "format" for s in spans)


def test_fluff_patterns() -> None:
    assert "summarize" in strip_fluff("Please just really summarize this").lower()
    assert "just" not in strip_fluff("Please just summarize").lower()
    out = strip_fluff("Hello, could you summarize cats? Thank you!!")
    assert "cats" in out
    assert "!!" not in out
