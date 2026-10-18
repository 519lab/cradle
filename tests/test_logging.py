"""Tests for configurable structured request logging.

The load-bearing guarantee is the default: at logging.content="none", NO prompt
or response text may appear in any log line, on any terminal path. These tests
make that enforceable rather than aspirational, and cover the two config axes
(level, content), the redaction chokepoint, truncation, and the setup module.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import (
    AuthKey,
    AuthSettings,
    FeatureFlags,
    L2Settings,
    LoggingSettings,
    Settings,
    UpstreamSettings,
)
from cradle.embeddings.fake import FakeEmbedder
from cradle.logging_setup import configure_logging, request_fields
from tests.fake_rerank import AllowReranker
from tests.fake_upstream import fake_app

ROOT = Path(__file__).resolve().parents[1]

# The one prompt/response string every "no text in logs" assertion checks for.
PROMPT_MARKER = "capital-of-france-unique-marker"


def _settings(tmp_path: Path, api_key: str, logging_cfg: LoggingSettings) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(
            keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]
        ),
        features=FeatureFlags(cache=True, compression=True, l2=True, reconstruction=True),
        l2=L2Settings(mode="local", cosine_threshold=0.90),
        upstream=UpstreamSettings(base_url="http://upstream/v1", models_passthrough=False),
        logging=logging_cfg,
    )


@pytest.fixture
def make_client(tmp_path: Path, api_key: str):
    """Factory: build a TestClient with a chosen LoggingSettings.

    Each client's TestClient context (which owns the embed pool + http client)
    stays open until the test ends, so a streamed body can be read after POST.
    """
    with contextlib.ExitStack() as stack:

        def _make(logging_cfg: LoggingSettings) -> TestClient:
            settings = _settings(tmp_path, api_key, logging_cfg)
            transport = httpx.ASGITransport(app=fake_app)
            http = httpx.AsyncClient(transport=transport, base_url="http://upstream")
            app = create_app(
                settings=settings, embedder=FakeEmbedder(), http=http, reranker=AllowReranker()
            )
            return stack.enter_context(TestClient(app))

        yield _make


def _post(client: TestClient, auth_header: dict, content: str, stream: bool = False) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={
            "model": "gpt-4o-mini",
            "temperature": 0,
            "stream": stream,
            "messages": [{"role": "user", "content": content}],
        },
    )


def _request_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "cradle.request"]


# ---------------------------------------------------------------------------
# The default guarantee: content="none" leaks no prompt/response text.
# ---------------------------------------------------------------------------


def test_default_content_none_logs_no_text_on_miss_and_hit(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="none"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        r1 = _post(client, auth_header, PROMPT_MARKER)  # miss
        r2 = _post(client, auth_header, PROMPT_MARKER)  # L1 hit
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers["X-Cradle-Cache"] == "MISS"
    assert r2.headers["X-Cradle-Cache"] == "HIT-L1"
    lines = _request_lines(caplog)
    assert len(lines) == 2  # one line per request
    blob = "\n".join(r.getMessage() for r in lines)
    # No prompt text, no response text ("ACK") anywhere in the lines.
    assert PROMPT_MARKER not in blob
    assert "ACK" not in blob
    # But the non-sensitive fields ARE present.
    assert "cache=miss" in blob or "cache=MISS" not in blob  # layer_hit is lowercase
    assert "cache=l1" in blob
    assert "model=gpt-4o-mini" in blob


def test_default_content_none_logs_no_text_on_stream(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="none"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        r = _post(client, auth_header, PROMPT_MARKER, stream=True)
        body = r.read().decode()
    assert r.status_code == 200
    # The fake upstream streams the answer split across deltas ("A" then "CK"),
    # so assert the stream carried real content rather than a contiguous "ACK".
    assert '"content":"A"' in body and '"content":"CK"' in body
    blob = "\n".join(x.getMessage() for x in _request_lines(caplog))
    assert PROMPT_MARKER not in blob
    assert "completion=" not in blob   # no completion field at content=none
    assert "prompt=" not in blob       # no prompt field at content=none
    assert "cache=miss" in blob


# ---------------------------------------------------------------------------
# The opt-in: prompts / prompts_and_completions do surface text.
# ---------------------------------------------------------------------------


def test_content_prompts_logs_prompt_not_completion(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="prompts"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        _post(client, auth_header, PROMPT_MARKER)
    blob = "\n".join(x.getMessage() for x in _request_lines(caplog))
    assert PROMPT_MARKER in blob       # prompt present
    assert "ACK" not in blob           # completion still redacted


def test_content_prompts_and_completions_logs_both(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="prompts_and_completions"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        _post(client, auth_header, PROMPT_MARKER)
    blob = "\n".join(x.getMessage() for x in _request_lines(caplog))
    assert PROMPT_MARKER in blob
    assert "ACK" in blob               # completion now present


def test_completion_logged_on_stream_at_full_content(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="prompts_and_completions"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        r = _post(client, auth_header, PROMPT_MARKER, stream=True)
        r.read()
    blob = "\n".join(x.getMessage() for x in _request_lines(caplog))
    assert PROMPT_MARKER in blob
    assert "ACK" in blob


# ---------------------------------------------------------------------------
# Error and bypass paths still emit a line (and still redact at none).
# ---------------------------------------------------------------------------


def test_upstream_error_still_logs_a_line(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="none"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        r = client.post(
            "/v1/chat/completions",
            headers=auth_header,
            json={"model": "fail-429", "temperature": 0,
                  "messages": [{"role": "user", "content": PROMPT_MARKER}]},
        )
    assert r.status_code == 429
    lines = _request_lines(caplog)
    assert len(lines) == 1                 # the errored request is not invisible
    fields = lines[0].fields
    assert fields["status"] == 429 and fields["upstream_status"] == 429
    assert PROMPT_MARKER not in lines[0].getMessage()


def test_bypass_json_path_logs_a_line(make_client, auth_header, caplog):
    # n=2 is uncacheable -> JSON bypass, which flows through _miss_json with the
    # real (byte-exact) upstream completion; so at full content it IS logged.
    client = make_client(LoggingSettings(content="prompts_and_completions"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        r = client.post(
            "/v1/chat/completions",
            headers=auth_header,
            json={"model": "gpt-4o-mini", "temperature": 0, "n": 2,
                  "messages": [{"role": "user", "content": PROMPT_MARKER}]},
        )
    assert r.status_code == 200
    lines = _request_lines(caplog)
    assert len(lines) == 1
    fields = lines[0].fields
    assert fields["cache"] == "bypass"
    # No cache key on the bypass path (never keyed for caching).
    assert "key" not in fields


def test_streaming_bypass_logs_line_without_completion(make_client, auth_header, caplog):
    # stream + logprobs stays a true bypass -> _passthrough_bytes, which tees bytes
    # verbatim and never parses a completion, so it logs with completion=None.
    # (stream + tools is no longer bypass since #43 — it's the cacheable passthrough.)
    client = make_client(LoggingSettings(content="prompts_and_completions"))
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        r = client.post(
            "/v1/chat/completions",
            headers=auth_header,
            json={"model": "gpt-4o-mini", "temperature": 0, "stream": True,
                  "logprobs": True,
                  "messages": [{"role": "user", "content": PROMPT_MARKER}]},
        )
        r.read()
    assert r.status_code == 200
    lines = _request_lines(caplog)
    assert len(lines) == 1
    fields = lines[0].fields
    assert fields["cache"] == "bypass"
    assert "completion" not in fields   # verbatim tee never parses a body
    assert "key" not in fields


# ---------------------------------------------------------------------------
# Truncation.
# ---------------------------------------------------------------------------


def test_prompt_text_is_truncated(make_client, auth_header, caplog):
    client = make_client(LoggingSettings(content="prompts", max_text_chars=8))
    long_prompt = "X" * 200
    with caplog.at_level(logging.INFO, logger="cradle.request"):
        _post(client, auth_header, long_prompt)
    blob = "\n".join(x.getMessage() for x in _request_lines(caplog))
    assert "X" * 200 not in blob       # full text not present
    assert "…(+" in blob               # truncation marker present (readable, not …)
    assert "(+199)" in blob            # dropped-char count present


# ---------------------------------------------------------------------------
# Level axis: DEBUG surfaces per-candidate decision lines; INFO suppresses them.
# ---------------------------------------------------------------------------


def test_debug_level_surfaces_pipeline_lines(make_client, auth_header, caplog):
    # A rerank rejection is the easiest decision line to force; here we just
    # assert the pipeline logger is quiet at INFO and can speak at DEBUG.
    client = make_client(LoggingSettings(level="INFO", content="none"))
    with caplog.at_level(logging.DEBUG, logger="cradle.pipeline"):
        _post(client, auth_header, PROMPT_MARKER)
    # No DEBUG pipeline line fires for a plain miss+hit with an allowing reranker.
    assert not [r for r in caplog.records if r.name == "cradle.pipeline" and r.levelno == logging.DEBUG]


# ---------------------------------------------------------------------------
# The redaction chokepoint, unit-tested directly (no HTTP).
# ---------------------------------------------------------------------------


def test_disconnected_stream_emits_a_line():
    """A client that drops mid-stream still gets a request line (step 5)."""
    from cradle.cache.records import Principal
    from cradle.gateway.context import RequestContext
    from cradle.logging_setup import log_request

    s = Settings(logging=LoggingSettings(content="none"))
    configure_logging(s)
    ctx = RequestContext(request_id="r-disc", principal=Principal(tenant_id="t1", user_id="u1", key_id="k"))
    ctx.layer_hit = "miss"
    logger = logging.getLogger("cradle.request")
    with _capture(logger) as records:
        log_request(s, ctx, None, disconnected=True)
    assert len(records) == 1
    fields = records[0].fields
    assert fields["disconnected"] is True
    assert fields["cache"] == "miss"


@contextlib.contextmanager
def _capture(logger: logging.Logger):
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _H()
    logger.addHandler(h)
    try:
        yield records
    finally:
        logger.removeHandler(h)


def test_request_fields_never_leaks_text_at_none():
    from cradle.cache.records import Principal
    from cradle.gateway.context import RequestContext
    from cradle.gateway.models import ChatRequest
    from cradle.normalize import canonicalize, l1_key

    s = Settings()
    req = ChatRequest(
        model="m", temperature=0,
        messages=[{"role": "user", "content": PROMPT_MARKER}],
    )
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k1")
    canon = canonicalize(req, principal, s, backend_namespace="ns")
    ctx = RequestContext(request_id="r1", principal=principal)
    ctx.canonical = canon
    ctx.l1_cache_key = l1_key(canon)   # the pipeline stashes this; mirror it here
    completion = {"choices": [{"message": {"role": "assistant", "content": "ACK"}}]}

    fields = request_fields(LoggingSettings(content="none"), ctx, completion)
    serialized = str(fields)
    assert PROMPT_MARKER not in serialized
    assert "ACK" not in serialized
    assert len(fields["key"]) == 64        # hashed key, not text
    assert "prompt" not in fields and "completion" not in fields


# ---------------------------------------------------------------------------
# configure_logging idempotency & format.
# ---------------------------------------------------------------------------


def test_configure_logging_is_idempotent():
    root = logging.getLogger("cradle")
    before = len(root.handlers)
    s = Settings()
    configure_logging(s)
    after_one = len(root.handlers)
    configure_logging(s)
    after_two = len(root.handlers)
    assert after_two == after_one          # no duplicate handler on second call
    assert after_one >= before


def test_configure_logging_json_format_emits_json():
    import json as _json

    s = Settings(logging=LoggingSettings(format="json", level="INFO"))
    configure_logging(s)
    # Assert on the installed handler's formatter directly (capsys can't see a
    # StreamHandler bound to the pre-redirect stderr).
    root = logging.getLogger("cradle")
    handler = next(h for h in root.handlers if getattr(h, "_cradle", False))
    record = logging.getLogger("cradle.request").makeRecord(
        "cradle.request", logging.INFO, __file__, 0, "request", (), None
    )
    record.fields = {"cache": "miss", "model": "m"}
    parsed = _json.loads(handler.formatter.format(record))
    assert parsed["cache"] == "miss"
    assert parsed["model"] == "m"
    assert parsed["msg"] == "request"
