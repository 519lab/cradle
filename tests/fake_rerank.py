"""Fake rerankers for tests, so no real cross-encoder model is loaded.

``AllowReranker`` returns a high score for everything (every candidate passes),
which keeps existing L2 tests unchanged: rerank is on by default but never
alters their behavior. Tests that want to exercise a rerank *rejection* pass a
``ScriptedReranker`` mapping candidate substrings to scores.
"""

from __future__ import annotations


class AllowReranker:
    """Always scores above any sane threshold; every candidate survives rerank."""

    def score(self, query: str, candidate: str) -> float:
        return 100.0


class ScriptedReranker:
    """Scores by the first matching (substring -> score) rule; default 100.0."""

    def __init__(self, rules: list[tuple[str, float]]) -> None:
        self._rules = rules

    def score(self, query: str, candidate: str) -> float:
        text = f"{query} || {candidate}"
        for needle, value in self._rules:
            if needle in text:
                return value
        return 100.0
