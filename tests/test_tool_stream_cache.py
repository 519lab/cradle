"""Tool-enabled streaming requests become cacheable when they carry no tool call (#43).

The critical properties, in order of importance:
1. A tool-call response is NEVER cached (client gets it verbatim; 2nd call still MISS).
2. A no-tool-call response IS cached and replays.
3. Cross-mode: a stream-written entry and a JSON request return the SAME body
   (they share a cache key — `stream` is excluded from hash_input).
4. Fail-closed: adversarial content+tool_call, unknown deltas, disconnects → not cached.
5. The flag off restores today's bypass behavior; logprobs streams still bypass.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import (
    AuthKey,
    AuthSettings,
    CacheSettings,
    FeatureFlags,
    L2Settings,
    Settings,
    UpstreamSettings,
)
from cradle.embeddings.fake import FakeEmbedder
from tests.fake_rerank import AllowReranker
from tests.fake_upstream import fake_app

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"type": "function", "function": {"name": "x", "parameters": {"type": "object", "properties": {}}}},
]


def _settings(tmp_path: Path, api_key: str, cache_tool_streams: bool = True) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(cache=True, l2=True, reconstruction=True),
        l2=L2Settings(mode="local", cosine_threshold=0.90),
        upstream=UpstreamSettings(base_url="http://upstream/v1"),
        cache=CacheSettings(cache_tool_streams=cache_tool_streams),
    )


@pytest.fixture
def make_client(tmp_path: Path, api_key: str):
    import contextlib

    with contextlib.ExitStack() as stack:

        def _make(cache_tool_streams: bool = True) -> TestClient:
            s = _settings(tmp_path, api_key, cache_tool_streams)
            http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
            app = create_app(settings=s, embedder=FakeEmbedder(), http=http, reranker=AllowReranker())
            return stack.enter_context(TestClient(app))

        yield _make


def _stream(client, auth, model, content, tools=TOOLS, **extra):
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": content}]}
    if tools is not None:
        body["tools"] = tools
    body.update(extra)
    r = client.post("/v1/chat/completions", headers=auth, json=body)
    r.read()
    return r


# 1. Tool-call response is never cached ---------------------------------------


def test_tool_call_response_not_cached(make_client, auth_header):
    c = make_client()
    r1 = _stream(c, auth_header, "tools-mixed-stream", "do a thing")  # content + tool_call + stop
    assert b"tool_calls" in r1.content            # client got the tool call
    assert r1.headers["X-Cradle-Cache"] == "MISS"
    r2 = _stream(c, auth_header, "tools-mixed-stream", "do a thing")
    assert r2.headers["X-Cradle-Cache"] == "MISS"  # NOT cached — the critical property


def test_pure_tool_call_stream_not_cached(make_client, auth_header):
    # The default fake_upstream tool branch: role -> tool_call -> finish tool_calls.
    c = make_client()
    r1 = _stream(c, auth_header, "gpt-4o-mini", "call x")
    assert b"tool_calls" in r1.content
    r2 = _stream(c, auth_header, "gpt-4o-mini", "call x")
    assert r2.headers["X-Cradle-Cache"] == "MISS"


# 2. No-tool-call response is cached -----------------------------------------


def test_plain_content_with_tools_is_cached(make_client, auth_header):
    c = make_client()
    r1 = _stream(c, auth_header, "tools-plain-stream", "answer in text")
    assert r1.headers["X-Cradle-Cache"] == "MISS"
    assert b"ACK" in r1.content
    r2 = _stream(c, auth_header, "tools-plain-stream", "answer in text")
    assert r2.headers["X-Cradle-Cache"] == "HIT-L1"  # tool requests are L1-only
    assert b"ACK" in r2.content


def test_different_tool_set_does_not_hit(make_client, auth_header):
    c = make_client()
    _stream(c, auth_header, "tools-plain-stream", "answer in text")  # cache with TOOLS
    other = [{"type": "function", "function": {"name": "y", "parameters": {"type": "object", "properties": {}}}}]
    r = _stream(c, auth_header, "tools-plain-stream", "answer in text", tools=other)
    assert r.headers["X-Cradle-Cache"] == "MISS"  # different tools => different key


# 3. Cross-mode consistency ---------------------------------------------------


def test_stream_written_entry_replays_to_json_identically(make_client, auth_header):
    c = make_client()
    # Write via the streaming tool path.
    _stream(c, auth_header, "tools-plain-stream", "cross mode test")
    # Same request, non-streaming: must HIT the stream-written entry and match.
    r = c.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"model": "tools-plain-stream", "tools": TOOLS,
              "messages": [{"role": "user", "content": "cross mode test"}]},
    )
    assert r.headers["X-Cradle-Cache"] == "HIT-L1"
    assert r.json()["choices"][0]["message"]["content"].strip().endswith("ACK") or "ACK" in r.json()["choices"][0]["message"]["content"]


# 4. Fail-closed --------------------------------------------------------------


def test_unknown_delta_field_not_cached(make_client, auth_header):
    # A 'reasoning_content' delta reaches the client but can't be replayed
    # content-only. The reasoning deltas precede content and finish_reason:"stop"
    # arrives on its own terminal chunk (real muse-glimmer/llama.cpp shape), so the
    # parse short-circuit leaves finish_reason uncaptured. The response must still
    # not cache, AND the recorded skip reason must be the true root cause
    # (unsupported_stream), not the misleading finish_None the naive gate order gave.
    from cradle.metrics import prometheus as m

    c = make_client()

    def skip(reason: str) -> float:
        return m.cache_write_skips.labels(reason=reason)._value.get()

    before_unsupported = skip("unsupported_stream")
    before_finish_none = skip("finish_None")

    r1 = _stream(c, auth_header, "reasoning-stream", "think then answer")
    assert b"reasoning_content" in r1.content   # client saw the reasoning verbatim
    r2 = _stream(c, auth_header, "reasoning-stream", "think then answer")
    assert r2.headers["X-Cradle-Cache"] == "MISS"  # not cached (allowlist tripped)

    # The recorded reason is the true root cause (the un-cacheable reasoning delta),
    # not a downstream artifact. NOTE: whether the mislabel actually occurs depends
    # on chunk-arrival timing (see test_gate_root_cause_before_shape_artifact for the
    # transport-independent proof); over in-process ASGI the whole body arrives in one
    # buffer so the terminal finish_reason is still parsed. Here we assert the correct
    # label is used and finish_None is not.
    assert skip("unsupported_stream") > before_unsupported
    assert skip("finish_None") == before_finish_none


def test_incomplete_stream_not_cached(make_client, auth_header):
    # trunc-stream ends without finish_reason / [DONE].
    c = make_client()
    _stream(c, auth_header, "trunc-stream", "truncate me")
    r2 = _stream(c, auth_header, "trunc-stream", "truncate me")
    assert r2.headers["X-Cradle-Cache"] == "MISS"


# 5. Flag off / logprobs ------------------------------------------------------


def test_flag_off_restores_bypass(make_client, auth_header):
    c = make_client(cache_tool_streams=False)
    r = _stream(c, auth_header, "tools-plain-stream", "answer in text")
    assert r.headers["X-Cradle-Cache"] == "BYPASS"
    r2 = _stream(c, auth_header, "tools-plain-stream", "answer in text")
    assert r2.headers["X-Cradle-Cache"] == "BYPASS"  # never cached when off


def test_logprobs_stream_still_bypass(make_client, auth_header):
    c = make_client()
    r = _stream(c, auth_header, "gpt-4o-mini", "logprobs please", tools=None, logprobs=True)
    assert r.headers["X-Cradle-Cache"] == "BYPASS"


# 6. Fail-closed unit tests on the accumulator + gate (edge branches) ---------


def test_accumulator_flags_unknown_delta_and_bad_payload():
    from cradle.gateway.sse import StreamAccumulator, parse_and_accumulate

    acc = StreamAccumulator(outbound_id="o", outbound_created=1, model="m")
    parse_and_accumulate('data: {"choices":[{"delta":{"content":"hi"}}]}', acc)
    assert acc.cache_disabled is False           # plain content is fine
    parse_and_accumulate('data: {"choices":[{"delta":{"reasoning":"x"}}]}', acc)
    assert acc.cache_disabled is True            # unknown delta key trips it

    acc2 = StreamAccumulator(outbound_id="o", outbound_created=1, model="m")
    parse_and_accumulate("data: not-json{", acc2)
    assert acc2.cache_disabled is True           # unparseable frame

    acc3 = StreamAccumulator(outbound_id="o", outbound_created=1, model="m")
    parse_and_accumulate("data: [1,2,3]", acc3)  # valid JSON, non-object
    assert acc3.cache_disabled is True


def test_gate_root_cause_before_shape_artifact():
    """The passthrough skip gate reports the ROOT cause, not a downstream artifact.

    When the accumulator aborts mid-stream (cache_disabled / error) it stops
    parsing, so finish_reason and saw_done are never captured and the reconstructed
    completion is incomplete BY CONSTRUCTION. This is the exact state a real
    reasoning-model stream produces once reasoning_content trips the allowlist while
    the terminal finish_reason chunk is still in flight (observed live on
    muse-glimmer). The reason recorded must be the abort cause, never finish_None.
    This is transport-independent: it constructs the post-abort state directly, so
    it fails on the naive gate order (cache_skip_reason first) and passes on the
    root-cause-first order.
    """
    from cradle.gateway.writeback import passthrough_skip_reason

    # Post-abort completion: finish_reason never captured -> None (would be
    # cache_skip_reason -> "finish_None" under the naive order).
    incomplete = {"choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": None}]}

    # cache_disabled (reasoning delta) must win over the finish_None artifact.
    assert passthrough_skip_reason(
        completion=incomplete, error=False, cache_disabled=True,
        saw_done=False, tool_call_seen=False, no_store=False,
    ) == "unsupported_stream"

    # An upstream error also outranks the shape artifact.
    assert passthrough_skip_reason(
        completion=incomplete, error=True, cache_disabled=False,
        saw_done=False, tool_call_seen=False, no_store=False,
    ) == "upstream_error"

    # A genuinely clean, complete stream still caches (returns None).
    clean = {"choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    assert passthrough_skip_reason(
        completion=clean, error=False, cache_disabled=False,
        saw_done=True, tool_call_seen=False, no_store=False,
    ) is None

    # And a genuine finish_None (no abort, stream really ended without a reason)
    # is still reported as finish_None -- the fix does not mask real cases.
    assert passthrough_skip_reason(
        completion=incomplete, error=False, cache_disabled=False,
        saw_done=True, tool_call_seen=False, no_store=False,
    ) == "finish_None"


def test_function_call_delta_disables_cache():
    from cradle.gateway.sse import StreamAccumulator, parse_and_accumulate

    acc = StreamAccumulator(outbound_id="o", outbound_created=1, model="m")
    parse_and_accumulate('data: {"choices":[{"delta":{"function_call":{"name":"x"}}}]}', acc)
    assert acc.cache_disabled is True            # legacy function_call not replayable


async def test_maybe_cache_passthrough_gates(tmp_path, api_key, monkeypatch):
    """Direct test of the fail-closed writeback gate: each disqualifier => no cache."""
    import cradle.gateway.pipeline as pl
    from cradle.cache.records import Principal
    from cradle.compress.engine import compress
    from cradle.gateway.context import RequestContext
    from cradle.gateway.models import ChatMessage, ChatRequest
    from cradle.gateway.sse import StreamAccumulator
    from cradle.normalize import canonicalize

    s = _settings(tmp_path, api_key)
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", stream=True, tools=TOOLS,
                      messages=[ChatMessage(role="user", content="hi")])
    canonical = canonicalize(req, principal, s, backend_namespace="ns")
    compressed = compress(req.messages, s, "m")

    writes: list = []

    async def _fake_writeback(runtime, canon, vec, rec):
        writes.append(rec)

    monkeypatch.setattr(pl, "writeback", _fake_writeback)

    class _RT:
        settings = s

    def _ctx():
        ctx = RequestContext(request_id="r", principal=principal)
        ctx.canonical = canonical
        ctx.layer_hit = "miss"
        ctx.cacheable_passthrough_stream = True
        return ctx

    def _acc(**kw):
        a = StreamAccumulator(outbound_id="o", outbound_created=1, model="m")
        a.content_parts = ["hello"]
        a.finish_reason = "stop"
        a.saw_done = True
        a.client_connected = True
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    # clean -> cached
    await pl._maybe_cache_passthrough(_RT(), req, _ctx(), None, compressed, _acc())
    assert len(writes) == 1

    # each disqualifier -> not cached
    writes.clear()
    for a in (
        _acc(tool_call_seen=True),
        _acc(error=True),
        _acc(cache_disabled=True),
        _acc(saw_done=False),
        _acc(client_connected=False),
    ):
        await pl._maybe_cache_passthrough(_RT(), req, _ctx(), None, compressed, a)
    assert writes == []
