from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cradle.cache.records import CanonicalRequest, Principal


@dataclass
class RequestContext:
    request_id: str
    principal: Principal
    canonical: CanonicalRequest | None = None
    # The L1 cache key (l1_key(canonical)), stashed by the pipeline when it
    # computes it so the request log line can report it without a second full
    # hash of the prompt body on the hot path. None on the bypass path.
    l1_cache_key: str | None = None
    layer_hit: Literal["l1", "l2", "miss", "bypass"] = "miss"
    l2_score: float | None = None
    # Set when an L2 candidate passed the cosine gate but the precision guard
    # rejected it (issue #5): "numbers" / "negation" / "no-text". Surfaced as
    # X-Cradle-Guard so a guard-forced miss is observable, not silent.
    l2_guard_reason: str | None = None
    # Rerank stage outcome, surfaced as X-Cradle-Rerank: the score for a served
    # hit ("pass:<score>"), a rejection ("reject:<score>"), or "fail-open" when
    # the reranker was unavailable and the candidate was served anyway.
    l2_rerank_note: str | None = None
    inbound_prompt_tokens: int = 0
    upstream_prompt_tokens: int = 0
    # Prompt tokens after rule-based compression, counted with Cradle's own
    # tokenizer (the same one as inbound), so inbound - compressed is the TRUE
    # compression saving under a single accounting. None until the miss path runs
    # compress(): compression never runs on hits or bypass, and comparing this
    # against upstream_prompt_tokens (a different backend tokenizer that also
    # includes the chat template) mixes two accountings and can read as negative
    # even though compression only ever removes tokens from the payload (#52).
    compressed_prompt_tokens: int | None = None
    t_l1_s: float = 0.0
    t_embed_s: float = 0.0
    t_l2_s: float = 0.0
    t_compress_s: float = 0.0
    t_upstream_s: float = 0.0
    t_reconstruct_s: float = 0.0
    # Raw client request headers (ASGI bytes pairs, repeats kept) relayed upstream
    # minus the denylist in upstream/openai.py (#81).
    client_headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    upstream_name: str = "default"
    # Per-request cache directives (enhancement #2), parsed from
    # X-Cradle-Cache-Control (no-store / no-cache / refresh) and X-Cradle-Cache-TTL.
    cache_no_read: bool = False   # skip L1/L2 lookup (no-cache / refresh)
    cache_no_store: bool = False  # skip writeback (no-store)
    cache_ttl_override: int | None = None
    # Probe mode (X-Cradle-Cache-Control: probe): run the read-side decision
    # only and return an explanation; never write, never call upstream. The
    # pipeline appends one entry per examined L2 candidate to probe_candidates.
    cache_probe: bool = False
    probe_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Volatility guard outcome (cache/volatility.py): the reason a prompt was
    # classified time-sensitive and given the short TTL. Surfaced as
    # X-Cradle-Volatile so a shortened TTL is observable, not silent.
    volatile_reason: str | None = None
    # Verified L2 (gateway/audit.py): this L2 hit was sampled for a background
    # audit against a fresh upstream answer. Surfaced as X-Cradle-Audit.
    audit_scheduled: bool = False
    # A cacheable stream+tools miss (#43): the response is teed to the client
    # verbatim (so a tool call relays intact) while a copy is accumulated, then
    # cached only if no tool call occurred. layer_hit stays "miss"; this flag is
    # the orthogonal wire strategy that routes to _passthrough_cache_stream.
    cacheable_passthrough_stream: bool = False
    # Single-flight (#57): when this request is the leader of a flight, the pipeline
    # attaches the Flight here so the miss path (_wrap_stream / _miss_json) can
    # publish its outbound frames / final completion to waiting followers and
    # resolve-or-fail the flight in a finally. None for a non-leader request.
    flight: Any = None
