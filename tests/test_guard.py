"""Regression fixture for the L2 precision guard (issue #5).

Each pair is the two role-framed embed_text strings Cradle actually embeds
("user: {content}\n").

The guard promises two exact signals: numbers and negation. Those are the
near-miss classes with the highest cosine (0.9421, 0.9847), which a threshold
can never catch. Entity swaps are a documented gap (see guard.py) that needs a
cross-encoder; they are covered by ``ENTITY_GAP`` below, asserted as NOT caught
so the gap is explicit and any future change that closes it is noticed here.
"""

from __future__ import annotations

import pytest

from cradle.cache.guard import guard_reason


def _framed(content: str) -> str:
    return f"user: {content}\n"


# Genuine paraphrases: guard must ALLOW (guard_reason is None).
MUST_ALLOW = [
    ("paraphrase-capital",
     "What is the capital of France? Answer in one word.",
     "Name France's capital city in a single word."),
    ("paraphrase-boil",
     "At what temperature in Celsius does water boil at sea level? Number only.",
     "Give just the number: water's boiling point in Celsius at sea level."),
    ("paraphrase-largest",
     "Which planet in our solar system is the largest? One word.",
     "What's the biggest planet in the solar system? Single word."),
]

# Near-misses the guard MUST reject, with the expected reason.
MUST_REJECT = [
    ("numeric-swap-usd", "numbers",
     "Convert 100 USD to EUR at a rate of 0.9. Number only.",
     "Convert 200 USD to EUR at a rate of 0.9. Number only."),
    ("negation-python", "negation",
     "Is Python a statically typed language? Answer yes or no.",
     "Is Python NOT a statically typed language? Answer yes or no."),
]

# Documented gap: entity swaps are NOT caught by numbers+negation. Asserted so
# the limitation is explicit and a future cross-encoder change trips this test.
ENTITY_GAP = [
    ("entity-swap-capital",
     "What is the capital of France? Answer in one word.",
     "What is the capital of Germany? Answer in one word."),
    ("entity-swap-planet",
     "Which planet is closest to the Sun? One word.",
     "Which planet is farthest from the Sun? One word."),
]


@pytest.mark.parametrize("label,query,candidate", MUST_ALLOW, ids=[p[0] for p in MUST_ALLOW])
def test_paraphrases_allowed(label: str, query: str, candidate: str) -> None:
    reason = guard_reason(_framed(query), _framed(candidate))
    assert reason is None, f"{label}: genuine paraphrase wrongly rejected ({reason})"


@pytest.mark.parametrize(
    "label,expected,query,candidate", MUST_REJECT, ids=[p[0] for p in MUST_REJECT]
)
def test_nearmiss_rejected(label: str, expected: str, query: str, candidate: str) -> None:
    assert guard_reason(_framed(query), _framed(candidate)) == expected, label


@pytest.mark.parametrize("label,query,candidate", ENTITY_GAP, ids=[p[0] for p in ENTITY_GAP])
def test_entity_swaps_are_a_known_gap(label: str, query: str, candidate: str) -> None:
    # numbers+negation cannot see entity swaps; this documents that, on purpose.
    assert guard_reason(_framed(query), _framed(candidate)) is None, label


def test_identical_text_allowed() -> None:
    t = _framed("What is the capital of France?")
    assert guard_reason(t, t) is None


def test_empty_candidate_rejected_no_text() -> None:
    """A pre-guard record with no persisted embed_text fails closed."""
    assert guard_reason(_framed("anything"), "") == "no-text"


def test_negation_contraction_detected() -> None:
    """n't contractions count as negation."""
    assert guard_reason(_framed("Is it typed?"), _framed("Isn't it typed?")) == "negation"


def test_decimal_and_grouped_numbers() -> None:
    assert guard_reason(_framed("pay 1,000 now"), _framed("pay 1000 now")) == "numbers"
    assert guard_reason(_framed("rate is 0.9"), _framed("rate is 0.8")) == "numbers"


def test_role_frame_not_counted() -> None:
    """Same content under identical framing is allowed."""
    assert guard_reason("user: what is water\n", "user: what is water\n") is None
