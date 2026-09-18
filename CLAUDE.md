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
docker compose -f docker-compose-ci.yml build
cp docker-compose-cpu.yml docker-compose.yml   # or docker-compose-gpu.yml; docker-compose.yml is gitignored
docker compose up --build
```

Compose templates: `docker-compose-cpu.yml` (default), `docker-compose-gpu.yml` (CUDA rerank), `docker-compose-ci.yml` (CI). Copy the one you want to `docker-compose.yml` (gitignored working copy) so plain `docker compose up` finds it.

Optional real-BGE pair eval (downloads the ONNX model):

```bash
uv run pytest -m embed
```

Property-based tests (`tests/test_properties.py`, Hypothesis) run in the default suite at 200 examples each; `CRADLE_HYPOTHESIS_MAX=5000 uv run pytest tests/test_properties.py` for a deeper local pass.

Noisy CI latency gate:

```bash
CRADLE_SKIP_LATENCY=1 uv run pytest
```

Default config: `config/cradle.yaml`. Override the fallback upstream with `CRADLE_UPSTREAM_BASE_URL`. Named backends live under `upstreams:` with `routes:` (`fnmatch` on `model`). Proxy listen is `127.0.0.1:8000`. Default is intercept mode: no Cradle API key; client `Authorization` is forwarded and used as the cache tenant. Optional `auth.keys` is an allowlist. OpenAI-compatible clients only; Claude Code’s Anthropic `/v1/messages` is not implemented.

Per-request cache directives come from `X-Cradle-Cache-Control` (`no-store`, `no-cache`/`refresh`, `probe`) and `X-Cradle-Cache-TTL`. `probe` is a dry run: it returns a `cradle.probe` JSON explanation of the L1/L2/guard/rerank decision and never writes or calls upstream (`src/cradle/gateway/probe.py`).

The volatility guard (`src/cradle/cache/volatility.py`, `cache.volatility_guard`) clamps the TTL of time-sensitive prompts to `cache.volatile_ttl_s`; it classifies user turns only, never system prompts, and an explicit `X-Cradle-Cache-TTL` overrides it.

Verified L2 (`src/cradle/gateway/audit.py`, `l2.audit_rate`, default off): a sampled L2 hit is re-asked upstream in a background task after the response; the verdict updates `cradle_l2_audit_total{verdict}`, the entry's `audit_floor` (enforced in the pipeline candidate loop), `{data_dir}/audits.jsonl`, and on disagree writes the fresh answer back under the query's key. Tasks are tracked on `Runtime.audit_tasks` and drained at shutdown.

L2 (Qdrant local) requires a **single** uvicorn worker. `WEB_CONCURRENCY` / `UVICORN_WORKERS` other than `1` is a startup error. Bypass streams (`stream+tools`, `n!=1`, logprobs) are raw SSE body passthrough (with an allowlist of upstream headers — `retry-after`, `x-ratelimit-*`, renamed request id — relayed); cacheable stream misses still wrap text completions. On a cacheable wrap-stream miss Cradle always requests `stream_options.include_usage` upstream (so real token usage is cached) even when the client did not; client-facing usage emission stays gated on the client's own flag. Upstream error bodies and `retry-after`/`x-ratelimit-*` headers are forwarded on errors too. Pass-through contract (both miss and audit paths): Cradle forwards only sampling fields the client actually set — `model_dump(exclude_unset=True, exclude_none=True)` — never injecting its own `ChatRequest` defaults (`temperature`/`top_p`/penalties/`n`/`stream`) onto a backend with its own; explicit values and vendor extras (`top_k`) are kept.

## Layout

`src/cradle/` — gateway, cache, embeddings, compress, reconstruct, upstream, metrics.
Modules stay ≤ 600 lines. No `ProxyService` god object.

Design contract: `DESIGN.md`. Product: `PRD.md`. Production operations: `RUNBOOK.md` (ADR-0002) — lock-step material: any change touching a config key, env var, metric name, health/readiness condition, capacity limit, or per-request header updates `RUNBOOK.md` in the same PR, and bumps its "Verified against" date when a command is re-checked against a running instance.
