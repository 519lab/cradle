"""Cache probe mode: explain what Cradle *would* do, without calling upstream.

``X-Cradle-Cache-Control: probe`` runs the full read-side decision pipeline
(canonicalize, L1 lookup, embed, L2 top-K, guard, rerank) and returns a JSON
explanation instead of a completion. Nothing is written: no L1 promote, no
writeback, no upstream call. Threshold tuning and "why did this hit?" questions
become a ``curl`` instead of a paid upstream round-trip and a YAML edit.

The pipeline records one entry per L2 candidate it examined in
``RequestContext.probe_candidates``; this module only shapes the response.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from cradle.gateway.context import RequestContext

PROBE_OBJECT = "cradle.probe"

_CACHE_LABEL = {"l1": "HIT-L1", "l2": "HIT-L2", "miss": "MISS", "bypass": "BYPASS"}


def candidate_entry(
    *,
    key: str,
    score: float,
    guard: str | None,
    rerank: str | None,
    served: bool,
) -> dict[str, Any]:
    """One examined L2 candidate, in the order the pipeline tried it."""
    return {
        "key": key,
        "cosine": round(score, 6),
        "guard": f"reject:{guard}" if guard else "pass",
        "rerank": rerank,
        "served": served,
    }


def probe_body(ctx: RequestContext, *, l1_key: str | None, l2_eligible: bool) -> dict[str, Any]:
    canonical = ctx.canonical
    return {
        "object": PROBE_OBJECT,
        "request_id": ctx.request_id,
        "cache": _CACHE_LABEL[ctx.layer_hit],
        "would_call_upstream": ctx.layer_hit in {"miss", "bypass"},
        "upstream": ctx.upstream_name,
        "pipeline_version": canonical.pipeline_version if canonical else "",
        "uncacheable_reason": canonical.uncacheable_reason if canonical else None,
        "inbound_prompt_tokens": ctx.inbound_prompt_tokens,
        "l1": {"key": l1_key, "hit": ctx.layer_hit == "l1"},
        "l2": {
            "eligible": l2_eligible,
            "embedded": ctx.t_embed_s > 0,
            "hit": ctx.layer_hit == "l2",
            "candidates": ctx.probe_candidates,
        },
    }


def probe_response(
    ctx: RequestContext,
    headers: dict[str, str],
    *,
    l1_key: str | None,
    l2_eligible: bool,
) -> JSONResponse:
    headers = {**headers, "X-Cradle-Probe": "1"}
    return JSONResponse(probe_body(ctx, l1_key=l1_key, l2_eligible=l2_eligible), headers=headers)
