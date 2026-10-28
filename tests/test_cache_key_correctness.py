"""Regression tests for the cache-key correctness bugs.

A: cache identity must include the resolved backend, so a routes change or two
   backends serving the same model glob never cross-replay answers.
B: routing-only hints (prompt_cache_key, ...) must NOT enter the cache key, so
   two identical prompts differing only by a hint share one L1 entry.
"""

from __future__ import annotations

from cradle.cache.records import Principal
from cradle.config import Settings
from cradle.gateway.models import ChatRequest
from cradle.normalize import cache_namespace, canonicalize, l1_key, sampling_fingerprint

P = Principal(tenant_id="t", user_id="u", key_id="k")
S = Settings()


def _req(**extra) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello there"}],
         "temperature": 0, **extra}
    )


# --- Bug A: backend namespace in the key ---------------------------------
def test_different_backend_yields_different_l1_key() -> None:
    ns_a = cache_namespace("openai", "https://api.openai.com/v1")
    ns_b = cache_namespace("local", "http://127.0.0.1:8080/v1")
    ka = l1_key(canonicalize(_req(), P, S, backend_namespace=ns_a))
    kb = l1_key(canonicalize(_req(), P, S, backend_namespace=ns_b))
    assert ka != kb, "same prompt on different backends must not share a cache key"


def test_same_backend_same_key() -> None:
    ns = cache_namespace("openai", "https://api.openai.com/v1")
    k1 = l1_key(canonicalize(_req(), P, S, backend_namespace=ns))
    k2 = l1_key(canonicalize(_req(), P, S, backend_namespace=ns))
    assert k1 == k2


def test_namespace_ignores_trailing_slash_and_case() -> None:
    assert cache_namespace("x", "http://H/v1/") == cache_namespace("x", "http://h/v1")


# --- Bug B: routing hints excluded from the key --------------------------
def test_prompt_cache_key_does_not_split_l1() -> None:
    base = l1_key(canonicalize(_req(), P, S))
    hinted = l1_key(canonicalize(_req(prompt_cache_key="tenant-42"), P, S))
    assert base == hinted, "prompt_cache_key is a routing hint; must not split the cache"


def test_multiple_routing_hints_ignored() -> None:
    base = l1_key(canonicalize(_req(), P, S))
    hinted = l1_key(canonicalize(
        _req(prompt_cache_key="a", prompt_cache_retention="24h", safety_identifier="sid"),
        P, S,
    ))
    assert base == hinted


# --- #82: session/chat identifiers and stored-completion tags excluded ------
_SESSION_FIELDS = {
    "session_id": "sess-1",
    "chat_id": "chat-1",
    "metadata": {"session_id": "meta-1"},
    "store": True,
}


def test_session_fields_do_not_split_l1_or_l2() -> None:
    base = canonicalize(_req(), P, S)
    for name, value in _SESSION_FIELDS.items():
        tagged = canonicalize(_req(**{name: value}), P, S)
        assert l1_key(base) == l1_key(tagged), name
        assert sampling_fingerprint(base) == sampling_fingerprint(tagged), name


def test_two_sessions_share_one_key() -> None:
    a = l1_key(canonicalize(_req(session_id="a", chat_id="c1", metadata={"x": 1}), P, S))
    b = l1_key(canonicalize(_req(session_id="b", chat_id="c2", metadata={"x": 2}), P, S))
    assert a == b


def test_vendor_sampling_extra_still_splits_with_session_fields() -> None:
    base = canonicalize(_req(session_id="s"), P, S)
    top_k = canonicalize(_req(session_id="s", top_k=40), P, S)
    assert l1_key(base) != l1_key(top_k)
    assert sampling_fingerprint(base) != sampling_fingerprint(top_k)


def test_real_param_still_splits() -> None:
    """A genuine generation-affecting extra must still change the key."""
    base = l1_key(canonicalize(_req(), P, S))
    real = l1_key(canonicalize(_req(some_unknown_sampling_param=0.5), P, S))
    assert base != real, "an undeclared non-hint field must still affect the key"
