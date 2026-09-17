from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.cache.records import Principal
from cradle.compress.structure import to_dense
from cradle.config import (
    AuthKey,
    AuthSettings,
    FeatureFlags,
    L2Settings,
    Settings,
    UpstreamSettings,
)
from cradle.embeddings.fake import FakeEmbedder
from cradle.gateway.models import ChatMessage, ChatRequest
from cradle.normalize import canonicalize, is_cacheable, l2_eligible
from cradle.tokens import count_chat_prompt, message_text
from tests.fake_upstream import fake_app


class ConstantEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 383

    def ready(self) -> bool:
        return True


def test_structure_identity() -> None:
    assert to_dense("hello", 400) == "hello"


def test_message_text_parts() -> None:
    m = ChatMessage(role="user", content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert message_text(m) == "a\nb"
    assert count_chat_prompt([ChatMessage(role="user", content="hi", name="n")], "gpt-4") > 3


def test_l2_eligible_and_non_text() -> None:
    s = Settings()
    p = Principal(tenant_id="t", user_id="u", key_id="k")
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
    c = canonicalize(req, p, s)
    assert l2_eligible(c, req, s) is True
    s2 = Settings(features=FeatureFlags(l2=False, cache=True))
    assert l2_eligible(c, req, s2) is False
    tools = ChatRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "x"}}],
    )
    ct = canonicalize(tools, p, s)
    assert l2_eligible(ct, tools, s) is False
    img = ChatRequest(
        model="m",
        messages=[ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "x"}}])],
    )
    ci = canonicalize(img, p, s)
    assert is_cacheable(ci, img, s) is False


def test_l2_promote_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_API_KEY", "test-key-aaaaaaaa")
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(cache=True, l2=True, compression=False, reconstruction=False),
        l2=L2Settings(mode="local"),
        upstream=UpstreamSettings(base_url="http://upstream/v1"),
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
    app = create_app(settings=settings, embedder=ConstantEmbedder(), http=http)
    headers = {"Authorization": "Bearer test-key-aaaaaaaa"}
    with TestClient(app) as c:
        a = c.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "m", "messages": [{"role": "user", "content": "alpha prompt"}]},
        )
        assert a.status_code == 200
        b = c.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "m", "messages": [{"role": "user", "content": "beta prompt"}]},
        )
        assert b.status_code == 200
        assert b.headers["X-Cradle-Cache"] in {"HIT-L2", "HIT-L1"}


def test_passthrough_reconstruction(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_API_KEY", "test-key-aaaaaaaa")
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(reconstruction=False, l2=False, cache=False, compression=False),
        upstream=UpstreamSettings(base_url="http://upstream/v1"),
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http)
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key-aaaaaaaa"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.headers["X-Cradle-Cache"] == "BYPASS"


def test_sse_usage_and_error_frames() -> None:
    from cradle.gateway.sse import error_frame, usage_frame

    assert error_frame("x")["error"]["code"] == "upstream_error"
    assert usage_frame("i", 1, "m", {"prompt_tokens": 1})["usage"]["prompt_tokens"] == 1


def test_merge_empty_and_tenant_template() -> None:
    from cradle.config import ReconstructSettings, ReconstructTenant, Settings
    from cradle.gateway.models import ReconstructionTemplate
    from cradle.reconstruct.merge import merge
    from cradle.reconstruct.templates import template_for

    assert merge({}, ReconstructionTemplate()) == {}
    s = Settings(
        reconstruct=ReconstructSettings(
            mode="wrap", tenants={"t1": ReconstructTenant(brand_prefix="P", brand_suffix="S")}
        )
    )
    t = template_for(s, "t1")
    assert t.brand_prefix == "P"


def test_l1_str_and_dict_get(tmp_path) -> None:
    from cradle.cache import l1 as l1mod
    from cradle.cache.records import CacheRecord

    settings = Settings(data_dir=tmp_path)
    cache = l1mod.open_l1(settings)
    rec = CacheRecord(
        key="k",
        tenant_id="t",
        user_id="u",
        model="m",
        system_prompt_version="none",
        pipeline_version="v1",
        prompt_hash="k",
        embed_text_hash="e",
        response={"choices": []},
        created_at=1,
        expires_at=9,
        ttl_s=8,
        inbound_prompt_tokens=1,
        upstream_prompt_tokens=0,
        sampling_fingerprint="s",
        temperature=1.0,
        top_p=1.0,
    )
    cache.set("skey", rec.model_dump_json(), expire=60, tag="v1", retry=True)
    assert l1mod.get_sync(cache, "skey") is not None
    cache.set("dkey", rec.model_dump(), expire=60, tag="v1", retry=True)
    assert l1mod.get_sync(cache, "dkey") is not None
    cache.close()


def test_n_not_one_uncacheable() -> None:
    p = Principal(tenant_id="t", user_id="u", key_id="k")
    s = Settings()
    req = ChatRequest(model="m", n=2, messages=[ChatMessage(role="user", content="hi")])
    c = canonicalize(req, p, s)
    assert is_cacheable(c, req, s) is False


def test_pass_through_client_auth(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.fake_upstream as fu

    monkeypatch.setenv("CRADLE_API_KEY", "test-key-aaaaaaaa")
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(l2=False, cache=False, compression=False, reconstruction=False),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http)
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key-aaaaaaaa"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
    assert fu.last_authorization == "Bearer test-key-aaaaaaaa"


def test_oversized_body(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cradle.config import ServerSettings

    monkeypatch.setenv("CRADLE_API_KEY", "test-key-aaaaaaaa")
    settings = Settings(
        data_dir=tmp_path / "data",
        server=ServerSettings(max_body_bytes=32),
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(l2=False, cache=False),
        upstream=UpstreamSettings(base_url="http://upstream/v1"),
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http)
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key-aaaaaaaa"},
            json={"model": "m", "messages": [{"role": "user", "content": "x" * 200}]},
        )
        assert r.status_code == 413


def test_upstream_connect_error_is_502(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_API_KEY", "test-key-aaaaaaaa")
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(l2=False, cache=False, compression=False, reconstruction=False),
        upstream=UpstreamSettings(base_url="http://127.0.0.1:9/v1", timeout_s=1),
    )
    app = create_app(settings=settings, embedder=FakeEmbedder())
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key-aaaaaaaa"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "upstream_error"


def test_multi_worker_refused(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_API_KEY", "test-key-aaaaaaaa")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(l2=True, cache=True),
        l2=L2Settings(mode="local"),
        upstream=UpstreamSettings(base_url="http://upstream/v1"),
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http)
    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY"):
        with TestClient(app):
            pass
