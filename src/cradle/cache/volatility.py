"""Volatility guard: spot time-sensitive prompts and shorten their cache TTL.

The most embarrassing semantic-cache failure is not a wrong paraphrase match
but a *correct* match replayed long after the answer changed: "what is the
latest version of X", "current price of Y", "today's date", "the weather in
Z". Cosine similarity, the guard and the reranker all agree those prompts
match — the answer is simply stale.

This module classifies a prompt's user text against a small pattern table and
returns a reason ("time", "market", "weather", "news") when the prompt asks
about something that changes on a timescale shorter than the default 24 h
TTL. The pipeline then clamps that entry's TTL to ``cache.volatile_ttl_s``
(0 = never store). An explicit client ``X-Cradle-Cache-TTL`` still wins: the
guard only fills in a default the client did not set.

False positives only shorten a TTL — they never change an answer — and every
classification is visible as ``X-Cradle-Volatile: <reason>``.

Patterns are deliberately conservative whole-word matches. Words that are
common in ordinary prompts ("now", "version", "score", "time") are excluded
unless anchored in a phrase ("right now", "what time is it").
"""

from __future__ import annotations

import re

from cradle.cache.records import CanonicalMessage

# (reason, compiled pattern). First match wins; order is by specificity.
VOLATILE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "time",
        re.compile(
            r"\b(today|tonight|tomorrow|yesterday|right now|currently|current|latest|newest"
            r"|most recent|recently|as of|this (week|month|year|morning|evening)"
            r"|at the moment|what time is it|what('s| is) the (date|time))\b",
            re.I,
        ),
    ),
    (
        "market",
        re.compile(
            r"\b(price|prices|pricing|stock price|share price|exchange rate|market cap"
            r"|how much (is|does|do) .{0,40}\b(cost|worth)|bitcoin|ethereum|crypto)\b",
            re.I,
        ),
    ),
    ("weather", re.compile(r"\b(weather|forecast|raining|snowing|humidity)\b", re.I)),
    ("news", re.compile(r"\b(news|headlines|breaking|trending|what happened)\b", re.I)),
)


def volatile_reason(text: str) -> str | None:
    """Return the volatility reason for a user prompt, or None if it looks stable."""
    for reason, pattern in VOLATILE_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def volatile_reason_for(messages: list[CanonicalMessage]) -> str | None:
    """Classify on user turns only.

    System/developer prompts routinely embed "today's date is ..." for the
    model's benefit; that must not mark every conversation volatile.
    """
    user_text = "\n".join(m.content for m in messages if m.role == "user")
    return volatile_reason(user_text) if user_text else None
