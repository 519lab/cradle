from __future__ import annotations

from cradle.compress.engine import compress
from cradle.config import FeatureFlags, Settings
from cradle.gateway.models import ChatMessage
from cradle.metrics import prometheus as m


def _over_compression_total() -> float:
    total = 0.0
    for metric in m.over_compression_blocks.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return total


def test_user_only_and_system_identity() -> None:
    s = Settings(features=FeatureFlags(compression=True, structure=False))
    messages = [
        ChatMessage(role="system", content="Please just really be careful"),
        ChatMessage(role="user", content="Please just really summarize cats"),
    ]
    out = compress(messages, s, "gpt-4o-mini")
    assert out.messages[0].content == "Please just really be careful"
    assert "just" not in (out.messages[1].content or "").lower()
    assert "summarize" in (out.messages[1].content or "")


def test_compression_off() -> None:
    s = Settings(features=FeatureFlags(compression=False))
    messages = [ChatMessage(role="user", content="Please just really summarize")]
    out = compress(messages, s, "gpt-4o-mini")
    assert out.savings_ratio == 0.0
    assert out.messages[0].content == "Please just really summarize"


def test_gate_skips_low_benefit_and_forwards_verbatim() -> None:
    # #52: a terse turn with no pleasantry/filler saves ~0 tokens. Below the
    # default floor, compression is a no-op: the ORIGINAL message object is
    # forwarded untouched and no reconstruction is set up.
    s = Settings(features=FeatureFlags(compression=True, structure=False))
    original = "Explain how binary search works on a sorted array."
    messages = [ChatMessage(role="user", content=original)]
    out = compress(messages, s, "gpt-4o-mini")
    assert out.savings_ratio == 0.0
    assert out.messages[0].content == original
    assert out.messages[0] is messages[0]  # same object, not a rewritten copy
    assert out.template.format_instructions == []


def test_gate_does_not_count_protected_spans_when_discarded() -> None:
    # A code prompt below the floor is forwarded verbatim; the discarded
    # compression must NOT inflate over_compression_blocks (the metric counts
    # protected spans only for compression that is actually used, #52).
    s = Settings(features=FeatureFlags(compression=True, structure=False))
    code = "```py\nx = 1\n```\nexplain this"
    before = _over_compression_total()
    out = compress([ChatMessage(role="user", content=code)], s, "gpt-4o-mini")
    after = _over_compression_total()
    assert out.savings_ratio == 0.0  # gated
    assert after == before


def test_gate_still_compresses_high_benefit() -> None:
    # A fluff-heavy turn clears the floor and is compressed as before.
    s = Settings(features=FeatureFlags(compression=True, structure=False))
    fluff = "Hi please just really could you summarize photosynthesis thanks thank you please"
    out = compress([ChatMessage(role="user", content=fluff)], s, "gpt-4o-mini")
    assert out.savings_ratio > 0.02
    assert "just" not in (out.messages[0].content or "").lower()


def test_gate_floor_is_a_threshold() -> None:
    # The floor is a real threshold on the achievable saving. This prompt saves a
    # measured ~0.176; a floor just below it compresses, a floor just above gates.
    fluff = "please just really simply explain the concept of recursion in programming"

    below = Settings(features=FeatureFlags(compression=True, structure=False))
    below.compress.min_savings_ratio = 0.10
    out_below = compress([ChatMessage(role="user", content=fluff)], below, "gpt-4o-mini")
    assert out_below.savings_ratio > 0.10
    assert "just" not in (out_below.messages[0].content or "").lower()

    above = Settings(features=FeatureFlags(compression=True, structure=False))
    above.compress.min_savings_ratio = 0.30
    out_above = compress([ChatMessage(role="user", content=fluff)], above, "gpt-4o-mini")
    assert out_above.savings_ratio == 0.0
    assert out_above.messages[0].content == fluff  # gated: forwarded verbatim


def test_gate_floor_high_disables_compression() -> None:
    # A floor above any achievable saving turns compression off per request.
    s = Settings(features=FeatureFlags(compression=True, structure=False))
    s.compress.min_savings_ratio = 1.0
    fluff = "Hi please just really could you summarize photosynthesis thanks thank you please"
    out = compress([ChatMessage(role="user", content=fluff)], s, "gpt-4o-mini")
    assert out.savings_ratio == 0.0
    assert out.messages[0].content == fluff
