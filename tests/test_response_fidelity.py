"""Upstream response data must reach the caller: token usage, provider metadata,
and real error detail — including on the wrap-mode streaming path, which rebuilds
the outbound stream rather than teeing bytes.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _sse_objects(body: bytes) -> list[dict]:
    objs: list[dict] = []
    for line in body.decode().splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        objs.append(json.loads(payload))
    return objs


def _usage_frames(objs: list[dict]) -> list[dict]:
    return [o for o in objs if o.get("usage") is not None]


def test_wrap_stream_live_usage_and_metadata(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    """A wrap-stream client that asks for usage gets the real upstream usage plus
    provider metadata (system_fingerprint, service_tier) on the usage chunk."""
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={
            "model": "gpt-4o-mini",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "fidelity-live-usage-1"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "MISS"
    frames = _usage_frames(_sse_objects(r.content))
    assert len(frames) == 1
    frame = frames[0]
    assert frame["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert frame["system_fingerprint"] == "fp_fake_123"
    assert frame["service_tier"] == "default"
    # DESIGN.md:148 — the outbound id is local, never the upstream chatcmpl-fake.
    assert frame["id"].startswith("chatcmpl-")
    assert frame["id"] != "chatcmpl-fake"


def test_wrap_stream_usage_survives_cache_roundtrip(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    """The headline bug: a first client that OMITS include_usage must not poison the
    cache. A later client that DOES request usage must get the real numbers, not {}."""
    body = {
        "model": "gpt-4o-mini",
        "stream": True,
        "messages": [{"role": "user", "content": "fidelity-roundtrip-usage-1"}],
    }
    # Request 1: no include_usage. Client sees no usage frame, but Cradle still asks
    # upstream and stores the true usage in the cache record.
    first = client.post("/v1/chat/completions", headers=auth_header, json=body)
    assert first.status_code == 200
    assert first.headers["X-Cradle-Cache"] == "MISS"
    assert _usage_frames(_sse_objects(first.content)) == []
    # A streaming miss learns the real upstream token count only after the body has
    # streamed — too late for a response header — so Cradle omits the header rather
    # than report a false 0. The truthful value lands in the cache record, asserted
    # via the include_usage replay below.
    assert "x-cradle-upstream-tokens" not in {k.lower() for k in first.headers}

    # Request 2: identical prompt, now WITH include_usage → L1 replay must carry usage.
    second = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={**body, "stream_options": {"include_usage": True}},
    )
    assert second.status_code == 200
    assert second.headers["X-Cradle-Cache"] == "HIT-L1"
    frames = _usage_frames(_sse_objects(second.content))
    assert len(frames) == 1
    assert frames[0]["usage"]["prompt_tokens"] == 10
    assert frames[0]["usage"]["total_tokens"] == 15


def test_wrap_stream_forwards_real_upstream_error(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    """A mid-stream upstream error object reaches the caller verbatim, not a generic
    'upstream error' string."""
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={
            "model": "err-stream",
            "stream": True,
            "messages": [{"role": "user", "content": "fidelity-error-1"}],
        },
    )
    assert r.status_code == 200
    objs = _sse_objects(r.content)
    errors = [o["error"] for o in objs if o.get("error")]
    assert len(errors) == 1
    assert errors[0]["message"] == "upstream boom"
    assert errors[0]["type"] == "rate_limit_error"
    assert errors[0]["code"] == "rate_limited"
    assert b"data: [DONE]\n\n" in r.content


def test_bypass_stream_payload_untouched(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    """The usage injection must never perturb the byte-exact bypass passthrough.
    A tools stream is bypass; upstream must not see an injected include_usage."""
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={
            "model": "gpt-4o-mini",
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
            "messages": [{"role": "user", "content": "bypass-untouched-1"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "BYPASS"
    # The fake upstream only emits a usage chunk when it received include_usage; a
    # bypass tools stream must not, proving the payload reached upstream unmodified.
    assert _usage_frames(_sse_objects(r.content)) == []
    assert b"tool_calls" in r.content
