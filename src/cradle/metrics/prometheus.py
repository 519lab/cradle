from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

inbound_prompt_tokens = Counter(
    "cradle_inbound_prompt_tokens_total", "Inbound prompt tokens", registry=REGISTRY
)
upstream_prompt_tokens = Counter(
    "cradle_upstream_prompt_tokens_total", "Upstream prompt tokens sent", registry=REGISTRY
)
cache_hits = Counter(
    "cradle_cache_hits_total", "Cache hits", ["layer"], registry=REGISTRY
)
cache_misses = Counter("cradle_cache_misses_total", "Cache misses", registry=REGISTRY)
embed_errors = Counter("cradle_embed_errors_total", "Embed errors", registry=REGISTRY)
over_compression_blocks = Counter(
    "cradle_over_compression_blocks_total",
    "Protected spans that blocked compression",
    ["reason"],
    registry=REGISTRY,
)
upstream_errors = Counter(
    "cradle_upstream_errors_total", "Upstream errors", ["status"], registry=REGISTRY
)
requests_total = Counter(
    "cradle_requests_total",
    "Requests",
    ["endpoint", "status", "cache"],
    registry=REGISTRY,
)
latency_seconds = Histogram(
    "cradle_latency_seconds",
    "Stage latency",
    ["stage"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5),
    registry=REGISTRY,
)
ready_gauge = Gauge("cradle_ready", "Component ready", ["component"], registry=REGISTRY)
l2_points = Gauge("cradle_l2_points", "L2 point count", registry=REGISTRY)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
