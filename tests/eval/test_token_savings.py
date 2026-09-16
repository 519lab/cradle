from __future__ import annotations

from pathlib import Path

from cradle.config import FeatureFlags, Settings
from cradle.eval.harness import load_jsonl, savings_for_row

GOLDEN = Path(__file__).parent / "golden_prompts.jsonl"


def test_golden_policy() -> None:
    rows = load_jsonl(GOLDEN)
    kinds = [r["kind"] for r in rows]
    assert kinds.count("verbose") >= 15
    assert kinds.count("dense") >= 5
    assert kinds.count("code") >= 5
    settings = Settings(features=FeatureFlags(compression=True, structure=False))
    ratios = []
    for row in rows:
        ratio = savings_for_row(row, settings)
        ratios.append(ratio)
        assert ratio + 1e-9 >= float(row["expect_min_savings"])
        if row["kind"] == "code":
            # fenced body still present after compress
            from cradle.compress.engine import compress
            from cradle.gateway.models import ChatMessage

            msgs = [ChatMessage.model_validate(m) for m in row["messages"]]
            out = compress(msgs, settings, "gpt-4o-mini")
            original = msgs[-1].content or ""
            compressed = out.messages[-1].content or ""
            if "```" in original:
                assert "```" in compressed
    mean = sum(ratios) / len(ratios)
    assert mean >= 0.40, mean
