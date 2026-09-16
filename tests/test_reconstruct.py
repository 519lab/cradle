from __future__ import annotations

from cradle.gateway.models import ReconstructionTemplate
from cradle.reconstruct.merge import merge, wrap_content, wrap_prefix, wrap_suffix


def test_always_prefix() -> None:
    t = ReconstructionTemplate(brand_prefix="PRE ", brand_suffix=" POST")
    assert wrap_prefix(t) == "PRE "
    body = wrap_content(t, "PRE hello")
    assert body.startswith("PRE ")
    assert body.endswith(" POST")


def test_format_in_suffix_if_missing() -> None:
    t = ReconstructionTemplate(format_instructions=["respond only in JSON"], brand_suffix="")
    assert "respond only in JSON" in wrap_suffix(t, "hello")
    assert wrap_suffix(t, "please respond only in JSON now") == ""


def test_merge_wraps_json() -> None:
    t = ReconstructionTemplate(brand_prefix="[", brand_suffix="]")
    out = merge(
        {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]},
        t,
    )
    assert out["choices"][0]["message"]["content"] == "[x]"
