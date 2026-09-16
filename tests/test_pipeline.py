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


def test_stream_tools_bypass(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "messages": [{"role": "user", "content": "call x"}],
    }
    r = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "BYPASS"
    assert b"tool_calls" in r.content
    assert b'"name":"x"' in r.content or b'"name": "x"' in r.content
    assert b"data: [DONE]\n\n" in r.content


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
