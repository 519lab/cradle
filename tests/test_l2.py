from __future__ import annotations

from cradle.cache import l2 as l2mod
from cradle.cache.records import CacheRecord, L2Filter
from cradle.config import Settings
from cradle.embeddings.fake import FakeEmbedder


def _rec(**kwargs) -> CacheRecord:
    base = dict(
        key="k1",
        tenant_id="t1",
        user_id="u1",
        model="m",
        system_prompt_version="none",
        pipeline_version="v1",
        prompt_hash="k1",
        embed_text_hash="e",
        response={"choices": [{"message": {"content": "ans"}}]},
        created_at=1,
        expires_at=9999999999,
        ttl_s=86400,
        inbound_prompt_tokens=1,
        upstream_prompt_tokens=1,
        sampling_fingerprint="fp1",
        temperature=0.0,
        top_p=1.0,
    )
    base.update(kwargs)
    return CacheRecord.model_validate(base)


def test_cross_tenant_no_hit(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    client = l2mod.open_l2(settings)
    emb = FakeEmbedder()
    vec = emb.embed("same prompt")
    rec = _rec()
    l2mod.upsert_sync(client, settings, vec, rec)
    hit = l2mod.query_sync(
        client,
        settings,
        vec,
        L2Filter(
            tenant_id="other",
            user_id="u1",
            model="m",
            backend_namespace="",
            system_prompt_version="none",
            pipeline_version="v1",
            sampling_fingerprint="fp1",
            now_unix=10,
        ),
    )
    assert hit is None
    client.close()


def test_cross_user_no_hit(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    client = l2mod.open_l2(settings)
    emb = FakeEmbedder()
    vec = emb.embed("same prompt")
    l2mod.upsert_sync(client, settings, vec, _rec())
    hit = l2mod.query_sync(
        client,
        settings,
        vec,
        L2Filter(
            tenant_id="t1",
            user_id="u2",
            model="m",
            backend_namespace="",
            system_prompt_version="none",
            pipeline_version="v1",
            sampling_fingerprint="fp1",
            now_unix=10,
        ),
    )
    assert hit is None
    client.close()


def test_sampling_fingerprint_mismatch(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    client = l2mod.open_l2(settings)
    vec = FakeEmbedder().embed("same")
    l2mod.upsert_sync(client, settings, vec, _rec(sampling_fingerprint="fp-a"))
    hit = l2mod.query_sync(
        client,
        settings,
        vec,
        L2Filter(
            tenant_id="t1",
            user_id="u1",
            model="m",
            backend_namespace="",
            system_prompt_version="none",
            pipeline_version="v1",
            sampling_fingerprint="fp-b",
            now_unix=10,
        ),
    )
    assert hit is None
    client.close()


def test_l2_hit_same_filters(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    client = l2mod.open_l2(settings)
    vec = FakeEmbedder().embed("hello")
    l2mod.upsert_sync(client, settings, vec, _rec())
    hit = l2mod.query_sync(
        client,
        settings,
        vec,
        L2Filter(
            tenant_id="t1",
            user_id="u1",
            model="m",
            backend_namespace="",
            system_prompt_version="none",
            pipeline_version="v1",
            sampling_fingerprint="fp1",
            now_unix=10,
        ),
    )
    assert hit is not None
    assert hit.score >= 0.90
    client.close()
