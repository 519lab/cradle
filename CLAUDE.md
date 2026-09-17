# Cradle

Pass-through API gateway for OpenAI-compatible LLM APIs: cache similar prompts, compress misses, forward the rest upstream.

## Branch model

- Long-lived: `develop` (integration) and `main` (blessed).
- Feature branches from `develop`. PRs target `develop`, never `main`.
- Never commit directly to `develop` or `main`.
- Conventional Commits with scope (`feat(cache):`, `fix(gateway):`).
- Changelog under `## [Unreleased]`.
- Remote: `https://github.com/519lab/cradle`. No PyPI in v1.

## Commands

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
uv run pytest --cov=cradle --cov-report=term-missing --cov-fail-under=90
uv run python -m cradle
docker compose -f compose.ci.yml build
docker compose up --build
```

Optional real-BGE pair eval (downloads the ONNX model):

```bash
uv run pytest -m embed
```

Noisy CI latency gate:

```bash
CRADLE_SKIP_LATENCY=1 uv run pytest
```

Default config: `config/cradle.yaml`. Override the fallback upstream with `CRADLE_UPSTREAM_BASE_URL`. Named backends live under `upstreams:` with `routes:` (`fnmatch` on `model`). Proxy listen is `127.0.0.1:8000`. Default is intercept mode: no Cradle API key; client `Authorization` is forwarded and used as the cache tenant. Optional `auth.keys` is an allowlist. OpenAI-compatible clients only; Claude Code’s Anthropic `/v1/messages` is not implemented.

Per-request cache directives come from `X-Cradle-Cache-Control` (`no-store`, `no-cache`/`refresh`, `probe`) and `X-Cradle-Cache-TTL`. `probe` is a dry run: it returns a `cradle.probe` JSON explanation of the L1/L2/guard/rerank decision and never writes or calls upstream (`src/cradle/gateway/probe.py`).

The volatility guard (`src/cradle/cache/volatility.py`, `cache.volatility_guard`) clamps the TTL of time-sensitive prompts to `cache.volatile_ttl_s`; it classifies user turns only, never system prompts, and an explicit `X-Cradle-Cache-TTL` overrides it.

L2 (Qdrant local) requires a **single** uvicorn worker. `WEB_CONCURRENCY` / `UVICORN_WORKERS` other than `1` is a startup error. Bypass streams (`stream+tools`, `n!=1`, logprobs) are raw SSE body passthrough (with an allowlist of upstream headers — `retry-after`, `x-ratelimit-*`, renamed request id — relayed); cacheable stream misses still wrap text completions. On a cacheable wrap-stream miss Cradle always requests `stream_options.include_usage` upstream (so real token usage is cached) even when the client did not; client-facing usage emission stays gated on the client's own flag. Upstream error bodies and `retry-after`/`x-ratelimit-*` headers are forwarded on errors too.

## Layout

`src/cradle/` — gateway, cache, embeddings, compress, reconstruct, upstream, metrics.
Modules stay ≤ 600 lines. No `ProxyService` god object.

Design contract: `DESIGN.md`. Product: `PRD.md`.
