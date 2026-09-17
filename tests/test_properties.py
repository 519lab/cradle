"""Property-based tests (innovation #6) for the invariant-heavy core: the
protected-span extractor, the fluff rules, the canonicalizer / cache key, and
the SSE synthesize→parse round trip.

Example tests pin behaviour on hand-picked inputs; these pin the *rules* on
thousands of generated ones. The alphabet is deliberately small and biased
toward the characters the guards care about (fences, ticks, braces, digits,
pleasantries, whitespace) so examples actually reach the guard branches.

Run harder locally with ``CRADLE_HYPOTHESIS_MAX=5000 uv run pytest tests/test_properties.py``.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cradle.cache.records import Principal
from cradle.compress.guards import extract_protected, restore_protected
from cradle.compress.rules import strip_fluff
from cradle.config import Settings
from cradle.gateway.models import ChatRequest
from cradle.gateway.sse import StreamAccumulator, parse_and_accumulate, synthesize_sse
from cradle.gateway.writeback import record_from
from cradle.normalize import _canonicalize_text, canonicalize, l1_key, sampling_fingerprint

settings.register_profile(
    "cradle",
    max_examples=int(os.environ.get("CRADLE_HYPOTHESIS_MAX", "200")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("cradle")

_ATOMS = st.sampled_from(
    list("ab {}[]`~\n\t\"':,.!0123456789")
    + ["```", "```json\n", "~~~", "please ", "thanks", "just ", "really ", "respond only in json\n"]
)
guard_texts = st.lists(_ATOMS, min_size=0, max_size=40).map("".join)

_WORDS = st.sampled_from(["alpha", "beta", "gamma", "42", "x1", "please", "thanks", "very"])
_WS = st.sampled_from([" ", "  ", "\t", "\n", " \n ", "　"])
plain_words = st.lists(_WORDS, min_size=1, max_size=12)


@pytest.fixture(scope="module")
def cfg() -> Settings:
    return Settings()


_PRINCIPAL = Principal(tenant_id="t", user_id="u", key_id="k")


# --- protected spans -------------------------------------------------------


@given(guard_texts)
def test_extract_restore_round_trips_any_text(text: str) -> None:
    placeholders, masked, _spans = extract_protected(text)
    assert restore_protected(masked, placeholders) == text


@given(guard_texts)
def test_protected_spans_are_disjoint_and_in_order(text: str) -> None:
    _placeholders, _masked, spans = extract_protected(text)
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.end <= b.start, (a, b)
    for s in spans:
        assert text[s.start : s.end] == s.text


@given(guard_texts)
def test_protected_spans_survive_fluff_stripping(text: str) -> None:
    placeholders, masked, spans = extract_protected(text)
    out = restore_protected(strip_fluff(masked), placeholders)
    for s in spans:
        assert s.text in out


# --- fluff rules -------------------------------------------------------------


@given(guard_texts)
def test_strip_fluff_is_idempotent(text: str) -> None:
    once = strip_fluff(text)
    assert strip_fluff(once) == once


@given(guard_texts)
def test_strip_fluff_never_grows_text(text: str) -> None:
    assert len(strip_fluff(text)) <= len(" ".join(text.split()))


# --- canonical text ----------------------------------------------------------


@given(st.lists(st.tuples(_WORDS, _WS, _WS), min_size=1, max_size=12))
def test_canonical_text_ignores_whitespace_outside_protected_spans(
    rows: list[tuple[str, str, str]],
) -> None:
    words = [w for w, _a, _b in rows]
    a = "".join(w + ws for w, ws, _ in rows)
    b = "".join(w + ws for w, _, ws in rows)
    assert _canonicalize_text(a) == _canonicalize_text(b) == " ".join(words)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known non-fixed-point: whitespace collapse joins the line before an unclosed "
        "fence onto the opener, so the fence is no longer at a line start on a second "
        "pass. Nothing re-canonicalizes canonical text (keys and embed_text are derived "
        "once), so this is documented rather than fixed; fixing it changes L1 keys for "
        "fenced prompts and needs a HASH_SCHEMA_VERSION bump."
    ),
)
def test_canonical_text_is_a_fixed_point_known_counterexample() -> None:
    text = "a\n```\n"
    once = _canonicalize_text(text)
    assert _canonicalize_text(once) == once


# --- cache identity ----------------------------------------------------------


_EXTRAS = st.dictionaries(
    st.sampled_from(["foo", "bar", "baz", "qux"]),
    st.one_of(st.integers(-5, 5), st.text(alphabet="xyz", max_size=3), st.booleans()),
    max_size=4,
)


def _req(body: dict) -> ChatRequest:
    return ChatRequest.model_validate(body)


@given(
    extras=_EXTRAS,
    words=st.lists(_WORDS, min_size=1, max_size=5),
    temperature=st.floats(0, 2).map(lambda v: round(v, 3)),
)
def test_l1_key_is_insensitive_to_field_order_and_empty_tools(
    extras: dict, words: list[str], temperature: float, cfg: Settings
) -> None:
    content = " ".join(words)
    base = {"model": "m", "temperature": temperature,
            "messages": [{"role": "user", "content": content}], **extras}
    reordered = dict(reversed(list(base.items())))
    k1 = l1_key(canonicalize(_req({**base, "tools": []}), _PRINCIPAL, cfg))
    k2 = l1_key(canonicalize(_req({**reordered, "tools": None}), _PRINCIPAL, cfg))
    k3 = l1_key(canonicalize(_req(reordered), _PRINCIPAL, cfg))
    assert k1 == k2 == k3
    # Any one extra changing must change the key.
    if extras:
        name = next(iter(extras))
        bumped = {**base, name: "changed-value"}
        assert l1_key(canonicalize(_req(bumped), _PRINCIPAL, cfg)) != k1


@given(
    words_a=st.lists(_WORDS, min_size=1, max_size=5),
    words_b=st.lists(_WORDS, min_size=1, max_size=5),
)
def test_sampling_fingerprint_ignores_messages_but_not_sampling(
    words_a: list[str], words_b: list[str], cfg: Settings
) -> None:
    a = canonicalize(_req({"model": "m", "messages": [{"role": "user", "content": " ".join(words_a)}]}),
                     _PRINCIPAL, cfg)
    b = canonicalize(_req({"model": "m", "messages": [{"role": "user", "content": " ".join(words_b)}]}),
                     _PRINCIPAL, cfg)
    assert sampling_fingerprint(a) == sampling_fingerprint(b)
    hot = canonicalize(
        _req({"model": "m", "temperature": 0.5,
              "messages": [{"role": "user", "content": " ".join(words_a)}]}),
        _PRINCIPAL, cfg,
    )
    assert sampling_fingerprint(hot) != sampling_fingerprint(a)


# --- SSE round trip -----------------------------------------------------------


@given(body=st.text(min_size=1, max_size=200), include_usage=st.booleans())
def test_synthesized_stream_parses_back_to_the_stored_body(
    body: str, include_usage: bool, cfg: Settings
) -> None:
    canonical = canonicalize(
        _req({"model": "m", "messages": [{"role": "user", "content": "q"}]}), _PRINCIPAL, cfg
    )
    completion = {
        "id": "chatcmpl-p", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": body},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    record = record_from(canonical, completion, 1, 1, 60)
    acc = StreamAccumulator(outbound_id="x", outbound_created=1, model="m")
    frames = b"".join(synthesize_sse(record, include_usage=include_usage))
    for line in frames.decode().split("\n"):
        parse_and_accumulate(line, acc)
    assert acc.content == body
    assert acc.finish_reason == "stop"
    assert acc.saw_done is True
    assert (acc.usage is not None) == include_usage
    assert frames.endswith(b"data: [DONE]\n\n")
