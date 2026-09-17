"""Verified L2 (innovation #1): audit-sampled hits, measured false-hit rate,
per-entry learned floors, and self-heal.

The prompt embedder is constant (every prompt is a cosine-1.0 L2 candidate,
so an entity swap "France" -> "Germany" is served as a wrong hit, exactly the
failure class the audit exists to catch). The scripted reranker passes every
prompt pair but scores the answer pair paris/berlin as contradictory, standing
in for the real cross-encoder's measured behaviour; answers embed by content
hash (FakeEmbedder) for the embed-judge fallback test. The fake upstream
answers by question, counting its calls.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.cache import l2 as l2mod
from cradle.config import AuthSettings, FeatureFlags, L2Settings, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder
from cradle.gateway.audit import AUDIT_LOG_NAME, cosine, should_audit
from cradle.metrics import prometheus as m
from tests.fake_rerank import ScriptedReranker

# Prompt pairs ("user: ..." texts) never match these rules and pass at 100;
# the answer pair paris/berlin (either order) scores as a contradiction.
_JUDGE_RULES = [("berlin || paris", -5.0), ("paris || berlin", -5.0)]

_CALLS = {"upstream": 0}


class PromptCollidingEmbedder:
    """Role-framed prompt text -> one constant vector; anything else (answers)
    -> FakeEmbedder's content-hash vector."""

    dim = 384

    def __init__(self) -> None:
        self._fake = FakeEmbedder()

    def embed(self, text: str) -> list[float]:
        if text.startswith(("user:", "system:", "developer:")):
            v = [0.0] * self.dim
            v[0] = 1.0
            return v
        return self._fake.embed(text)

    def ready(self) -> bool:
        return True


async def _upstream(request: httpx.Request) -> httpx.Response:
    _CALLS["upstream"] += 1
    body = request.content.decode().lower()
    if "fail-audit" in body:
        return httpx.Response(500, json={"error": {"message": "boom"}})
    answer = "berlin" if "germany" in body else "paris" if "france" in body else "other"
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{answer}", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )


def _make_client(tmp_path: Path, *, rerank: bool = True, **l2: object) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, l2_rerank=rerank, reconstruction=False, local_1b=False,
        ),
        l2=L2Settings(**{"mode": "local", "cosine_threshold": 0.90, "audit_rate": 1.0, **l2}),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(_upstream), base_url="http://upstream")
    return TestClient(create_app(
        settings=settings,
        embedder=PromptCollidingEmbedder(),
        http=http,
        reranker=ScriptedReranker(_JUDGE_RULES) if rerank else None,
    ))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    _CALLS["upstream"] = 0
    with _make_client(tmp_path) as c:
        yield c


def _ask(c: TestClient, content: str) -> httpx.Response:
    return c.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer audit-tok"},
        json={"model": "qwen", "temperature": 0,
              "messages": [{"role": "user", "content": content}]},
    )


def _drain(c: TestClient, timeout_s: float = 5.0) -> None:
    """Wait for background audit tasks scheduled on the app loop."""
    rt = c.app.state.runtime
    deadline = time.monotonic() + timeout_s
    while rt.audit_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not rt.audit_tasks, "audit tasks did not finish"


def _verdicts() -> dict[str, float]:
    out = {}
    for sample in m.l2_audits.collect()[0].samples:
        if sample.name.endswith("_total"):
            out[sample.labels["verdict"]] = sample.value
    return out


def _l2_payload(c: TestClient, key: str) -> dict:
    rt = c.app.state.runtime
    pts = rt.qdrant.retrieve(rt.settings.l2.collection, ids=[l2mod.point_id(key)], with_payload=True)
    return dict(pts[0].payload)


def test_should_audit_respects_rate() -> None:
    assert should_audit(0.0) is False
    assert should_audit(1.0) is True


def test_cosine_basics() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_wrong_hit_is_caught_floored_and_self_healed(client: TestClient) -> None:
    before = _verdicts()
    seed = _ask(client, "what is the capital of France")
    assert seed.headers["X-Cradle-Cache"] == "MISS" and "paris" in seed.text

    # Entity swap collides on cosine (constant prompt embedder): served WRONG.
    wrong = _ask(client, "what is the capital of Germany")
    assert wrong.headers["X-Cradle-Cache"] == "HIT-L2"
    assert wrong.headers["X-Cradle-Audit"] == "scheduled"
    assert "paris" in wrong.text
    calls_at_serve = _CALLS["upstream"]

    _drain(client)
    assert _CALLS["upstream"] == calls_at_serve + 1  # exactly one audit call
    after = _verdicts()
    assert after.get("disagree", 0) - before.get("disagree", 0) == 1

    # The served entry learned a floor at the similarity that produced the wrong
    # hit; find it through the audit log row (hit_key = the seed's record).
    rt = client.app.state.runtime
    rows = [json.loads(line) for line in (rt.settings.data_dir / AUDIT_LOG_NAME).read_text().splitlines()]
    row = rows[-1]
    assert row["verdict"] == "disagree" and row["query_similarity"] == 1.0
    assert row["judge"] == "rerank" and row["answer_score"] == -5.0
    assert "query_text" not in row  # no prompt text by default
    seed_payload = _l2_payload(client, row["hit_key"])
    assert seed_payload["audit_floor"] == 1.0
    assert seed_payload["audit_disagree"] == 1

    # Self-heal: the querying prompt now has its own verified entry.
    healed = _ask(client, "what is the capital of Germany")
    assert healed.headers["X-Cradle-Cache"] == "HIT-L1"
    assert "berlin" in healed.text

    # A new paraphrase still collides with BOTH points; the seed refuses (floor)
    # and the healed Germany entry serves the right answer.
    para = _ask(client, "capital city of Germany?")
    assert para.headers["X-Cradle-Cache"] == "HIT-L2"
    assert "berlin" in para.text
    _drain(client)


def test_seed_refuses_at_floor_even_when_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With only the floored entry present, a colliding query is a real MISS."""
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    _CALLS["upstream"] = 0
    with _make_client(tmp_path) as c:
        _ask(c, "what is the capital of France")
        _ask(c, "what is the capital of Germany")  # wrong hit -> audited
        _drain(c)
        rt = c.app.state.runtime
        # Remove the healed Germany point + L1 so only the floored seed remains.
        rt.l1.clear()
        rows = [json.loads(x) for x in (rt.settings.data_dir / AUDIT_LOG_NAME).read_text().splitlines()]
        rt.qdrant.delete(rt.settings.l2.collection, points_selector=[l2mod.point_id(rows[-1]["query_key"])])
        r = _ask(c, "what is the capital of Spain")  # collides with the seed at 1.0 <= floor 1.0
        assert r.headers["X-Cradle-Cache"] == "MISS"
        assert r.headers["X-Cradle-Guard"] == "reject:audit-floor"
        assert "other" in r.text


def test_correct_hit_agrees_and_keeps_no_floor(client: TestClient) -> None:
    before = _verdicts()
    _ask(client, "what is the capital of France")
    ok = _ask(client, "which city is the capital of France")
    assert ok.headers["X-Cradle-Cache"] == "HIT-L2" and "paris" in ok.text
    _drain(client)
    after = _verdicts()
    assert after.get("agree", 0) - before.get("agree", 0) == 1
    rt = client.app.state.runtime
    rows = [json.loads(x) for x in (rt.settings.data_dir / AUDIT_LOG_NAME).read_text().splitlines()]
    assert rows[-1]["verdict"] == "agree" and rows[-1]["judge"] == "rerank"
    assert rows[-1]["answer_score"] == 100.0
    payload = _l2_payload(client, rows[-1]["hit_key"])
    assert payload["audit_floor"] is None and payload["audit_agree"] == 1


def test_embed_judge_fallback_when_rerank_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """audit_judge=auto without a reranker falls back to answer-embedding cosine."""
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    _CALLS["upstream"] = 0
    with _make_client(tmp_path, rerank=False) as c:
        _ask(c, "what is the capital of France")
        # Agree case first: once the wrong hit below is audited, the seed is
        # floored at 1.0 and can never serve a colliding query again.
        ok = _ask(c, "which city is the capital of France")
        assert ok.headers["X-Cradle-Cache"] == "HIT-L2" and "paris" in ok.text
        _drain(c)
        wrong = _ask(c, "what is the capital of Germany")
        assert wrong.headers["X-Cradle-Cache"] == "HIT-L2" and "paris" in wrong.text
        _drain(c)
        rows = [json.loads(x) for x in (c.app.state.runtime.settings.data_dir / AUDIT_LOG_NAME).read_text().splitlines()]
        by_verdict = {r["verdict"]: r for r in rows}
        assert by_verdict["disagree"]["judge"] == "embed"
        assert by_verdict["disagree"]["answer_score"] < 0.5   # hash vectors: paris vs berlin
        assert by_verdict["agree"]["answer_score"] == pytest.approx(1.0)


def test_rerank_judge_without_reranker_is_an_error_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    with _make_client(tmp_path, rerank=False, audit_judge="rerank") as c:
        _ask(c, "what is the capital of France")
        before = _verdicts()
        _ask(c, "which city is the capital of France")
        _drain(c)
        assert _verdicts().get("error", 0) - before.get("error", 0) == 1


def test_audit_judge_is_validated() -> None:
    with pytest.raises(ValueError):
        L2Settings(audit_judge="llm")
    with pytest.raises(ValueError):
        L2Settings(audit_rate=1.5)


def test_audit_upstream_error_is_counted_not_raised(client: TestClient) -> None:
    before = _verdicts()
    _ask(client, "what is the capital of France")
    r = _ask(client, "fail-audit please, capital of France")  # colliding hit; audit call 500s
    assert r.headers["X-Cradle-Cache"] == "HIT-L2"
    _drain(client)
    after = _verdicts()
    assert after.get("error", 0) - before.get("error", 0) == 1
    rt = client.app.state.runtime
    rows = [json.loads(x) for x in (rt.settings.data_dir / AUDIT_LOG_NAME).read_text().splitlines()]
    assert rows[-1]["verdict"] == "error"
    assert _l2_payload(client, rows[-1]["hit_key"])["audit_floor"] is None


def test_audit_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    assert Settings().l2.audit_rate == 0.0
    with _make_client(tmp_path, audit_rate=0.0) as c:
        _ask(c, "what is the capital of France")
        r = _ask(c, "which city is the capital of France")
        assert r.headers["X-Cradle-Cache"] == "HIT-L2"
        assert "X-Cradle-Audit" not in r.headers
        assert not c.app.state.runtime.audit_tasks


def test_audit_log_text_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    with _make_client(tmp_path, audit_log_text=True) as c:
        _ask(c, "what is the capital of France")
        _ask(c, "which city is the capital of France")
        _drain(c)
        rows = [json.loads(x) for x in (c.app.state.runtime.settings.data_dir / AUDIT_LOG_NAME).read_text().splitlines()]
        assert "capital of France" in rows[-1]["query_text"]
        assert "capital of France" in rows[-1]["candidate_text"]


def test_record_audit_on_missing_point_is_noop(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    client = l2mod.open_l2(settings)
    assert l2mod.record_audit_sync(client, settings, "nope", query_similarity=0.9, agree=False) is None
    client.close()
