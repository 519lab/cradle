from __future__ import annotations

from cradle.cache import l1 as l1mod
from cradle.cache.records import CacheRecord
from cradle.config import Settings


def _rec(key: str) -> CacheRecord:
    return CacheRecord(
        key=key,
        tenant_id="t1",
        user_id="u1",
        model="m",
        system_prompt_version="none",
        pipeline_version="v1",
        prompt_hash=key,
        embed_text_hash="e",
        response={"id": key, "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]},
        created_at=1,
        expires_at=9999999999,
        ttl_s=86400,
        inbound_prompt_tokens=1,
        upstream_prompt_tokens=0,
        sampling_fingerprint="s",
        temperature=1.0,
        top_p=1.0,
    )


def test_l1_roundtrip(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    cache = l1mod.open_l1(settings)
    rec = _rec("abc")
    l1mod.set_sync(cache, rec.key, rec, 86400, "v1")
    got = l1mod.get_sync(cache, rec.key)
    assert got is not None
    assert got.response["id"] == "abc"
    cache.close()
