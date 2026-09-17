"""Verified L2: audit a sample of served semantic hits against a fresh answer.

Cradle's promise is "a miss beats a wrong hit", yet nothing in production ever
measured how often an L2 hit was wrong. This module closes that loop, off the
critical path:

1. After an L2 hit is served, with probability ``l2.audit_rate`` a background
   task calls upstream with the same (compressed, wrapped) request a miss would
   have sent, and judges whether the fresh answer agrees with the served one.
2. The result is a labeled observation ``(query similarity, answer score,
   verdict)``. It is counted (``cradle_l2_audit_total{verdict}`` — a real
   false-hit rate on real traffic), optionally appended to
   ``{data_dir}/audits.jsonl`` for offline threshold calibration, and folded
   into the served entry's own state: an entry judged *wrong* at similarity
   ``s`` refuses future matches at or below ``s`` (``CacheRecord.audit_floor``,
   enforced in the pipeline candidate loop as ``X-Cradle-Guard:
   reject:audit-floor``).
3. A *disagree* also self-heals: the fresh answer is written back under the
   querying prompt's own key, so the next paraphrase hits the right entry.

This is a deliberately non-parametric cousin of vCache's per-entry learned
thresholds (arXiv 2502.03771): no sigmoid fit, no confidence bands, strictly
monotone and conservative, so it has no cold-start cliff.

The judge (``judge_answers``) is a seam with two local, free implementations,
chosen by ``l2.audit_judge``. Measured on the bge models Cradle ships with:
the cross-encoder reranker scores same-meaning answers >= 7.2 and
contradictory ones <= 2.1 (Mercury-closest vs Neptune-farthest: -0.46;
"dynamically typed" vs "statically typed": 2.10), so it is the default when
loaded. Bi-encoder cosine cannot separate them (same 0.83-0.98 overlaps wrong
0.73-0.92; the planet pair scores 0.87), so it is only a fallback with a high
threshold that prefers a false "disagree" (a miss) over a missed error. The
raw score and judge name are always logged.

Audits cost real upstream calls (bounded by ``audit_rate``) and reuse the
client's forwarded ``Authorization`` after its response has completed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any

from cradle.cache import l2 as l2mod
from cradle.cache.records import CacheRecord, CanonicalRequest, L2Hit
from cradle.compress.engine import compress
from cradle.config import UpstreamSettings
from cradle.gateway.context import RequestContext
from cradle.gateway.models import ChatRequest
from cradle.gateway.writeback import cache_skip_reason, record_from, writeback
from cradle.metrics import prometheus as m
from cradle.reconstruct.merge import merge
from cradle.reconstruct.templates import template_for
from cradle.upstream.openai import UpstreamError, chat

if TYPE_CHECKING:
    from cradle.runtime import Runtime

log = logging.getLogger("cradle.audit")

AUDIT_LOG_NAME = "audits.jsonl"


def should_audit(rate: float, rng: random.Random | None = None) -> bool:
    if rate <= 0.0:
        return False
    return (rng or random).random() < rate


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _content(completion: dict[str, Any]) -> str:
    choices = completion.get("choices") or [{}]
    return str(((choices[0] or {}).get("message") or {}).get("content") or "")


async def judge_answers(runtime: Runtime, fresh: str, cached: str) -> tuple[str, float, bool]:
    """The judge seam. Returns ``(judge, score, agree)``.

    ``rerank`` scores the pair with the loaded cross-encoder (raw logit,
    threshold ``audit_rerank_threshold``); ``embed`` uses bi-encoder cosine
    (threshold ``audit_embed_threshold``). ``auto`` picks rerank when loaded.
    """
    settings = runtime.settings.l2
    loop = asyncio.get_running_loop()
    use_rerank = settings.audit_judge == "rerank" or (
        settings.audit_judge == "auto" and runtime.reranker is not None
    )
    if use_rerank:
        if runtime.reranker is None:
            raise RuntimeError("l2.audit_judge=rerank but no reranker is loaded")
        score = await loop.run_in_executor(
            runtime.embed_pool, runtime.reranker.score, fresh, cached
        )
        return "rerank", float(score), float(score) >= settings.audit_rerank_threshold
    assert runtime.embedder is not None
    va = await loop.run_in_executor(runtime.embed_pool, runtime.embedder.embed, fresh)
    vb = await loop.run_in_executor(runtime.embed_pool, runtime.embedder.embed, cached)
    score = cosine(va, vb)
    return "embed", score, score >= settings.audit_embed_threshold


def _append_log(runtime: Runtime, row: dict[str, Any]) -> None:
    if not runtime.settings.l2.audit_log:
        return
    path = runtime.settings.data_dir / AUDIT_LOG_NAME
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        log.exception("audit log append failed")


async def run_audit(
    runtime: Runtime,
    *,
    req: ChatRequest,
    ctx: RequestContext,
    canonical: CanonicalRequest,
    query_vec: list[float] | None,
    hit: L2Hit,
    served: CacheRecord,
    target: UpstreamSettings,
) -> str:
    """Audit one served L2 hit. Returns the verdict: agree | disagree | error."""
    settings = runtime.settings
    compressed = compress(req.messages, settings, req.model)
    tmpl = template_for(settings, ctx.principal.tenant_id)
    compressed.template.brand_prefix = tmpl.brand_prefix
    compressed.template.brand_suffix = tmpl.brand_suffix
    compressed.template.mode = tmpl.mode
    # Same pass-through contract as the miss path (issue #26): the audit re-asks
    # upstream the request a miss would have sent, so it must forward only fields the
    # client set — not Cradle's sampling defaults.
    payload = req.model_dump(exclude_unset=True, exclude_none=True)
    payload["messages"] = [msg.model_dump(exclude_none=True) for msg in compressed.messages]
    payload.pop("stream", None)
    payload.pop("stream_options", None)

    row: dict[str, Any] = {
        "ts": int(time.time()),
        "request_id": ctx.request_id,
        "tenant_id": canonical.tenant_id,
        "query_key": served.key,
        "hit_key": hit.record.key,
        "query_similarity": round(hit.score, 6),
        "guard": ctx.l2_guard_reason,
        "rerank": ctx.l2_rerank_note,
    }
    if settings.l2.audit_log_text:
        row["query_text"] = canonical.embed_text
        row["candidate_text"] = hit.record.embed_text

    try:
        fresh_completion = await chat(
            runtime.http, target, payload, authorization=ctx.headers.get("authorization") or None
        )
        fresh = merge(fresh_completion, compressed.template)
        # The same write-quality gate as the miss path (issue #24): an empty,
        # truncated or refused fresh answer is not evidence about the served
        # one and must never become the cached answer through self-heal.
        skip = cache_skip_reason(fresh)
        if skip is not None:
            m.l2_audits.labels(verdict="error").inc()
            m.cache_write_skips.labels(reason=skip).inc()
            _append_log(runtime, {**row, "verdict": "error", "skip": skip})
            return "error"
        judge, score, agree = await judge_answers(
            runtime, _content(fresh), _content(served.response)
        )
    except (UpstreamError, Exception):  # noqa: BLE001 - an audit must never raise
        log.exception("audit failed for request %s", ctx.request_id)
        m.l2_audits.labels(verdict="error").inc()
        _append_log(runtime, {**row, "verdict": "error"})
        return "error"

    verdict = "agree" if agree else "disagree"
    m.l2_audits.labels(verdict=verdict).inc()
    m.l2_audit_answer_score.labels(judge=judge).observe(score)
    if runtime.qdrant is not None:
        await l2mod.record_audit(
            runtime.qdrant, settings, hit.record.key, query_similarity=hit.score, agree=agree
        )
    if not agree:
        # Self-heal: the querying prompt gets its own, verified entry (L1 overwrite
        # of the promoted copy; L2 point when the query vector is known).
        usage = fresh_completion.get("usage") or {}
        upstream_tokens = int(usage.get("prompt_tokens") or compressed.compressed_tokens)
        healed = record_from(
            canonical, fresh, ctx.inbound_prompt_tokens, upstream_tokens, served.ttl_s
        )
        await writeback(runtime, canonical, query_vec, healed)
    _append_log(
        runtime, {**row, "judge": judge, "answer_score": round(score, 6), "verdict": verdict}
    )
    return verdict


def schedule_audit(runtime: Runtime, **kwargs: Any) -> asyncio.Task[None]:
    """Fire-and-track an audit; the served response is never delayed by it."""

    async def _run() -> None:
        try:
            await run_audit(runtime, **kwargs)
        except Exception:  # noqa: BLE001 - background task; never propagate
            log.exception("audit task crashed")

    task = asyncio.create_task(_run())
    runtime.audit_tasks.add(task)
    task.add_done_callback(runtime.audit_tasks.discard)
    return task
