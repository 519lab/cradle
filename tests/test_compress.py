from __future__ import annotations

from cradle.compress.engine import compress
from cradle.config import FeatureFlags, Settings
from cradle.gateway.models import ChatMessage


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
