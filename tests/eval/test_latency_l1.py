from __future__ import annotations

import os
import statistics
import time

import pytest

from cradle.cache import l1 as l1mod
from cradle.cache.records import CacheRecord
from cradle.config import Settings

pytestmark = pytest.mark.skipif(os.environ.get("CRADLE_SKIP_LATENCY") == "1", reason="skip latency")


def _rec(i: int) -> CacheRecord:
    body = "x" * 1800
    return CacheRecord(
        key=f"k{i:04d}",
        tenant_id="t",
        user_id="u",
        model="m",
        system_prompt_version="none",
        pipeline_version="v1",
        prompt_hash=f"k{i:04d}",
        embed_text_hash="e",
        response={
            "id": f"id{i}",
            "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
        },
        created_at=1,
        expires_at=9999999999,
        ttl_s=86400,
        inbound_prompt_tokens=10,
        upstream_prompt_tokens=0,
        sampling_fingerprint="s",
        temperature=1.0,
        top_p=1.0,
    )


def test_l1_p99_under_2ms(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    cache = l1mod.open_l1(settings)
    for i in range(1000):
        rec = _rec(i)
        l1mod.set_sync(cache, rec.key, rec, 86400, "v1")
    keys = [f"k{i:04d}" for i in range(200)]
    for k in keys[:20]:
        l1mod.get_sync(cache, k)
    times = []
    for k in keys:
        t0 = time.perf_counter()
        rec = l1mod.get_sync(cache, k)
        times.append(time.perf_counter() - t0)
        assert rec is not None
    p99 = statistics.quantiles(times, n=100)[-1]
    cache.close()
    assert p99 < 0.002, p99
