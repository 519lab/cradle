"""L2 semantic-cache precision guard.

The bi-encoder L2 lookup has recall but no precision: cosine similarity alone
cannot tell "capital of France" from "capital of Germany" (0.9054), a 100 vs
200 USD conversion (0.9421), or "is Python static" vs "is Python NOT static"
(0.9847, since negation barely moves a bi-encoder). See issue #5.

This guard runs *after* the cosine gate passes and *before* an L2 candidate is
served. It compares the incoming query's ``embed_text`` against the stored
candidate's ``embed_text`` and rejects the hit on two exact, robust signals:

- numbers differ    -> different quantities, different answer
- negation differs  -> opposite question, opposite answer

These two cover precisely the near-miss classes a cosine threshold can never
catch, because they carry the *highest* similarity: a 100 vs 200 swap scores
0.9421 and an "is" vs "is NOT" flip scores 0.9847, both far above any threshold
that still admits genuine paraphrases.

KNOWN GAP: entity swaps ("capital of France" vs "Germany", "closest" vs
"farthest planet") are NOT caught here. A capitalized-token / content-word
heuristic was tried and rejected: it wrongly killed genuine paraphrases
(synonyms, inflection, possessives) while still missing lowercase swaps. Closing
that class needs a cross-encoder rerank (a v2 design decision), not a heuristic.

A rejection is treated as an L2 miss: the request falls through to the upstream
and writes back a fresh, correct entry. Failing closed (reject) is deliberate,
because a wrong cached answer is worse than a slow correct one.
"""

from __future__ import annotations

import re

# A "number" is any run of digits (optionally with decimal point / commas),
# so "100", "200", "3.14", "1,000" each compare as distinct tokens.
_NUMBER_RE = re.compile(r"\d[\d,.]*")

# Negation markers. "n't" attaches to a preceding word (isn't, don't, can't) so
# it is counted separately from the standalone words.
_NEGATION_WORDS = frozenset({"not", "no", "never", "none", "cannot", "without", "nor"})
_NT_RE = re.compile(r"n't\b")

# Role framing ("user:", "assistant:") is stripped before comparison so it never
# affects the number/negation counts.
_ROLE_PREFIX_RE = re.compile(r"(?im)^(system|developer|user|assistant|tool)\s*:\s*")


def _strip_role_prefixes(text: str) -> str:
    """Remove leading 'role:' framing so it does not count as content."""
    return _ROLE_PREFIX_RE.sub("", text)


def _numbers(text: str) -> frozenset[str]:
    return frozenset(m.group(0).rstrip(".,") for m in _NUMBER_RE.finditer(text))


def _negation_count(text: str) -> int:
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    count = sum(1 for w in words if w in _NEGATION_WORDS)
    count += len(_NT_RE.findall(lowered))
    return count


def guard_reason(query_text: str, candidate_text: str) -> str | None:
    """Return a rejection reason, or None if the candidate may be served.

    ``query_text`` and ``candidate_text`` are the two ``embed_text`` strings
    (role-framed, as stored). An empty candidate (a pre-guard record with no
    persisted ``embed_text``) is rejected as ``"no-text"`` so it self-heals on
    the next miss rather than serving an unverified hit.
    """
    if not candidate_text:
        return "no-text"

    q = _strip_role_prefixes(query_text)
    c = _strip_role_prefixes(candidate_text)

    if _numbers(q) != _numbers(c):
        return "numbers"

    # A different count of negation markers means the question was flipped.
    if _negation_count(q) != _negation_count(c):
        return "negation"

    return None
