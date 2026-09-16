from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from cradle.cache import l1 as l1mod
from cradle.cache import l2 as l2mod
from cradle.cache.records import CacheRecord, CanonicalRequest
from cradle.normalize import l1_key

if TYPE_CHECKING:
    from cradle.runtime import Runtime


def _embed_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_from(
    canonical: CanonicalRequest,
    response: dict[str, Any],
    inbound: int,
    upstream_tokens: int,
    ttl_s: int,
    *,
    l2_score: float | None = None,
) -> CacheRecord:
    now = int(time.time())
    key = l1_key(canonical)
    return CacheRecord(
        key=key,
        tenant_id=canonical.tenant_id,
        user_id=canonical.user_id,
        model=canonical.model,
        system_prompt_version=canonical.system_prompt_version,
        pipeline_version=canonical.pipeline_version,
        prompt_hash=key,
        embed_text_hash=_embed_hash(canonical.embed_text),
        response=response,
        created_at=now,
        expires_at=now + ttl_s,
        ttl_s=ttl_s,
        inbound_prompt_tokens=inbound,
        upstream_prompt_tokens=upstream_tokens,
        l2_score=l2_score,
        sampling_fingerprint=canonical.sampling_fingerprint,
        temperature=canonical.temperature,
        top_p=canonical.top_p,
        max_tokens=canonical.max_tokens,
        max_completion_tokens=canonical.max_completion_tokens,
        n=canonical.n,
        seed=canonical.seed,
        stop=canonical.stop,
        response_format=canonical.response_format,
        presence_penalty=canonical.presence_penalty,
        frequency_penalty=canonical.frequency_penalty,
    )


async def writeback(
    runtime: Runtime,
    canonical: CanonicalRequest,
    vec: list[float] | None,
    record: CacheRecord,
) -> None:
    if runtime.l1 is not None:
        await l1mod.set(
            runtime.l1,
            record.key,
            record,
            record.ttl_s,
            runtime.settings.pipeline_version,
        )
    if vec is not None and runtime.qdrant is not None and runtime.settings.features.l2:
        await l2mod.upsert(runtime.qdrant, runtime.settings, vec, record)


async def promote_l2_hit(
    runtime: Runtime,
    canonical: CanonicalRequest,
    hit_record: CacheRecord,
    inbound: int,
) -> CacheRecord:
    now = int(time.time())
    ttl = runtime.settings.cache.ttl_s
    key = l1_key(canonical)
    promoted = hit_record.model_copy(
        update={
            "key": key,
            "prompt_hash": key,
            "created_at": now,
            "expires_at": now + ttl,
            "ttl_s": ttl,
            "inbound_prompt_tokens": inbound,
            "upstream_prompt_tokens": 0,
        }
    )
    if runtime.l1 is not None:
        await l1mod.set(runtime.l1, key, promoted, ttl, runtime.settings.pipeline_version)
    return promoted
