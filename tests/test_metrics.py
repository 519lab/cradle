from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_after_chat(client: TestClient, auth_header: dict[str, str]) -> None:
    client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "metric-me"}]},
    )
    body = client.get("/metrics", headers=auth_header).text
    assert "cradle_inbound_prompt_tokens_total" in body
    assert "cradle_requests_total" in body
