from __future__ import annotations

from fastapi.testclient import TestClient


def test_l1_hit_after_miss(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "unique-l1-prompt-xyz"}],
    }
    a = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert a.status_code == 200
    assert a.headers["X-Cradle-Upstream"] == "default"
    assert a.headers["X-Cradle-Cache"] in {"MISS", "HIT-L2"}
    b = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert b.status_code == 200
    assert b.headers["X-Cradle-Cache"] == "HIT-L1"
    assert a.json()["choices"][0]["message"]["content"] == b.json()["choices"][0]["message"]["content"]


def test_stream_tools_relays_tool_call_and_does_not_cache(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # #43: stream+tools is now the cacheable passthrough path (MISS, not BYPASS).
    # The fake upstream emits a tool call for a tools request, so the client must
    # receive it intact and NOTHING may be cached (a 2nd identical call is MISS).
    payload = {
        "model": "gpt-4o-mini",
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "messages": [{"role": "user", "content": "call x"}],
    }
    r = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "MISS"
    assert b"tool_calls" in r.content  # tool call relayed verbatim
    assert b'"name":"x"' in r.content or b'"name": "x"' in r.content
    assert b"data: [DONE]\n\n" in r.content
    # Second identical call must still MISS — the tool-call response is not cached.
    r2 = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r2.headers["X-Cradle-Cache"] == "MISS"


def test_stream_connect_error_is_json(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"model": "fail-401", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401
    assert r.headers.get("content-type", "").startswith("application/json")
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_truncated_stream_not_fake_stop(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"model": "trunc-stream", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert b"finish_reason\":\"stop\"" not in r.content.replace(b" ", b"")
    assert b"upstream stream ended without finish_reason" in r.content


def test_n2_json_keeps_all_choices(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"model": "gpt-4o-mini", "n": 2, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "BYPASS"
    # #52: a bypass request flows through the miss path but is an uncacheable
    # passthrough — a compression saving is not meaningful, so the header is absent.
    assert "X-Cradle-Compressed-Tokens" not in r.headers
    choices = r.json()["choices"]
    assert len(choices) == 2
    assert choices[1]["message"]["content"] == "ALT"


def test_stream_miss_writeback(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "stream": True,
        "messages": [{"role": "user", "content": "wrap-stream-miss-1"}],
    }
    s = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert s.status_code == 200
    assert s.headers["X-Cradle-Cache"] == "MISS"
    # #52: the compressed-tokens header is present on a streaming miss too.
    assert "X-Cradle-Compressed-Tokens" in s.headers
    assert b'"content":"A"' in s.content or b'"content": "A"' in s.content
    assert b'"content":"CK"' in s.content or b'"content": "CK"' in s.content
    j = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={**payload, "stream": False},
    )
    assert j.headers["X-Cradle-Cache"] == "HIT-L1"
    assert j.json()["choices"][0]["message"]["content"] == "ACK"


def test_json_and_stream_same_stored_body(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "identical-canonical-body-1"}],
    }
    j = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert j.headers["X-Cradle-Cache"] in {"MISS", "HIT-L1", "HIT-L2"}
    stored = j.json()["choices"][0]["message"]["content"]
    s = client.post("/v1/chat/completions", headers=auth_header, json={**payload, "stream": True})
    assert s.status_code == 200
    assert s.headers["X-Cradle-Cache"] == "HIT-L1"
    assert b"data: [DONE]\n\n" in s.content
    # replay concatenates stored body
    assert stored == "ACK"
    assert "ACK" in s.text


def test_compressed_tokens_header_present_on_miss_and_not_above_inbound(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # #52: X-Cradle-Compressed-Tokens exposes the compressed size under Cradle's
    # OWN tokenizer, so inbound - compressed is the true saving. Compression only
    # ever removes tokens, so compressed <= inbound always. (The upstream count is
    # a different tokenizer + chat template and must not be used for this ratio.)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "explain-the-compressed-header-xyz"}],
    }
    r = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "MISS"
    assert "X-Cradle-Compressed-Tokens" in r.headers
    inbound = int(r.headers["X-Cradle-Inbound-Tokens"])
    compressed = int(r.headers["X-Cradle-Compressed-Tokens"])
    assert 0 < compressed <= inbound


def test_compressed_tokens_header_absent_on_hit(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # Compression runs only on the miss path. On a cache hit the header must be
    # ABSENT (not "0"), so it is never misread as "compressed to nothing" (#52).
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hit-has-no-compressed-header-xyz"}],
    }
    miss = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert miss.headers["X-Cradle-Cache"] == "MISS"
    assert "X-Cradle-Compressed-Tokens" in miss.headers
    hit = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert hit.headers["X-Cradle-Cache"] == "HIT-L1"
    assert "X-Cradle-Compressed-Tokens" not in hit.headers


def test_compressed_tokens_header_reflects_real_stripping(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # A prompt full of strippable fluff must show compressed < inbound, proving the
    # header reports compression's actual effect (not a constant).
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Hi, could you please just simply explain recursion, thanks!"}
        ],
    }
    r = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r.headers["X-Cradle-Cache"] == "MISS"
    inbound = int(r.headers["X-Cradle-Inbound-Tokens"])
    compressed = int(r.headers["X-Cradle-Compressed-Tokens"])
    assert compressed < inbound
