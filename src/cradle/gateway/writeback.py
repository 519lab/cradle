from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from cradle.cache import l1 as l1mod
from cradle.cache import l2 as l2mod
from cradle.cache.records import CacheRecord, CanonicalRequest
from cradle.metrics import prometheus as m
from cradle.normalize import l1_key

if TYPE_CHECKING:
    from cradle.runtime import Runtime

log = logging.getLogger("cradle.writeback")


def _embed_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# finish_reasons that indicate a complete, trustworthy answer worth caching.
# `length` (truncated), `content_filter` (refused), tool_calls, and empty bodies
# would otherwise become the permanent cached answer for a prompt and its
# paraphrases (enhancement #3, write-quality gate). Shared by the miss path
# and the audit self-heal (issue #24) so no path can cache what the other refuses.
CACHEABLE_FINISH = frozenset({"stop", "eos"})


def cache_skip_reason(completion: dict[str, Any]) -> str | None:
    """Return a skip reason if this response must NOT be cached, else None."""
    choices = completion.get("choices") or []
    if not choices:
        return "no_choices"
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish not in CACHEABLE_FINISH:
        return f"finish_{finish}"
    content = (choice.get("message") or {}).get("content")
    if not content or not content.strip():
        return "empty_content"
    return None


def passthrough_skip_reason(
    *,
    completion: dict[str, Any],
    error: bool,
    cache_disabled: bool,
    saw_done: bool,
    tool_call_seen: bool,
    no_store: bool,
) -> str | None:
    """Ordered skip-reason gate for the tool-stream passthrough path (#43).

    Order matters: the recorded reason must name the ROOT cause. When the stream
    accumulator aborts mid-stream (``error`` or ``cache_disabled``), it stops
    parsing, so ``finish_reason``/``saw_done`` are never captured and the
    reconstructed ``completion`` is incomplete *by construction*. Reading
    ``cache_skip_reason`` off it would then report a misleading ``finish_None`` /
    ``incomplete_stream`` that is only a downstream artifact of the abort. So the
    aborting reasons are checked BEFORE the shape gates that inspect the
    (necessarily incomplete) completion. Caching still requires every check to
    pass; only the *reported* reason changes. Returns None when cacheable.

    Motivated by a live observation: a reasoning model (llama.cpp muse-glimmer)
    emits ``reasoning_content`` deltas, tripping ``cache_disabled`` mid-stream;
    on the wire the stream still ends with ``finish_reason:"stop"``, yet the old
    ordering recorded ``finish_None`` because the terminal chunk arrived after the
    parse short-circuit and was never seen.
    """
    if error:
        return "upstream_error"
    if cache_disabled:
        return "unsupported_stream"
    shape = cache_skip_reason(completion)
    if shape is not None:
        return shape
    if no_store:
        return "no_store"
    if tool_call_seen:
        return "tool_call"
    if not saw_done:
        return "incomplete_stream"
    return None


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
        backend_namespace=canonical.backend_namespace,
        system_prompt_version=canonical.system_prompt_version,
        pipeline_version=canonical.pipeline_version,
        prompt_hash=key,
        embed_text_hash=_embed_hash(canonical.embed_text),
        embed_text=canonical.embed_text,
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
        # Keep the l2_points gauge fresh on write (issue #4). This is an
        # approximate bump: an upsert that updates an existing point over-counts
        # by one until the purge loop's count_points() reconciles the true value.
        m.l2_points.inc()


async def writeback_best_effort(
    runtime: Runtime,
    canonical: CanonicalRequest,
    vec: list[float] | None,
    record: CacheRecord,
) -> None:
    """Write to cache, treating a backend failure as non-fatal (#70).

    The cache write is the LAST step of a successful miss: the answer has already
    reached the client (and, on the single-flight path, already committed the
    flight to success — followers are replaying it). An L1/L2 backend error here
    must not turn that success into an error response, and on the stream path must
    not append a trailing error frame onto an already-completed SSE body. Count it
    and log the traceback (never silent — an error is never just log noise), then
    return normally. CancelledError propagates (Exception only)."""
    try:
        await writeback(runtime, canonical, vec, record)
    except Exception:
        m.cache_write_errors.inc()
        log.exception("cache writeback failed for key %s (answer already served)", record.key)


async def promote_l2_hit(
    runtime: Runtime,
    canonical: CanonicalRequest,
    hit_record: CacheRecord,
    inbound: int,
    *,
    ttl_s: int | None = None,
) -> CacheRecord:
    """Copy an L2 hit into L1 under the querying prompt's exact key.

    ``ttl_s`` is the effective TTL for *this* request (client override or
    volatility guard); it defaults to the configured TTL.
    """
    now = int(time.time())
    ttl = runtime.settings.cache.ttl_s if ttl_s is None else ttl_s
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
    if runtime.l1 is not None and ttl > 0:
        await l1mod.set(runtime.l1, key, promoted, ttl, runtime.settings.pipeline_version)
    return promoted
