# Changelog

## [Unreleased]

### Added

- L2 semantic-cache **precision guard**: before serving a semantic hit, Cradle now compares the query against the stored candidate and rejects the hit when their numbers or negation differ, then falls through to a real upstream call. This stops the cache from returning a stale wrong answer to a near-miss prompt — e.g. a "convert 200 USD" request no longer replays the "100 USD" answer, and "is X true?" no longer replays the answer to "is X NOT true?". A rejection is observable via the new `X-Cradle-Guard: reject:<reason>` response header and the `cradle_l2_guard_rejects_total{reason}` metric. Known gap: single-entity swaps (capital of France vs Germany) are not caught by this guard and need a cross-encoder (tracked in #5). (#5)
- v1 Cradle pass-through API gateway: OpenAI-compatible `/v1/chat/completions` with L1 diskcache, L2 Qdrant local + FastEmbed, rule-based compression, prefix/suffix reconstruction, Prometheus metrics, and a fixture eval harness. No live LLM required for tests.
- Runnable multi-stage Docker image (`uv sync --frozen`) and Compose stack. FastEmbed weights bake into `/opt/cradle/models/fastembed` so the `/data` volume does not hide them. Compose reaches a host-side upstream via `host.docker.internal`. CI builds `compose.ci.yml` and smokes `/healthz` + `/readyz`.

### Changed

- Product copy: Cradle is a gateway clients point at, not a local companion app for a model runtime.
- Default auth is intercept mode: no Cradle API key. Client `Authorization` is forwarded upstream; cache isolation is SHA-256 of that token. Optional `auth.keys` remains an allowlist.
- Model glob `routes` send a request to a named OpenAI-compatible `upstreams` entry (OpenAI, xAI, llama.cpp, LiteLLM). Unmatched models use `upstream.base_url`.

### Fixed

- Bypass streams (`stream+tools`, `n!=1`, logprobs) now forward upstream SSE bytes instead of stripping everything but `delta.content`.
- Upstream stream HTTP errors return JSON OpenAI errors before SSE starts.
- Truncated streams no longer fake `finish_reason=stop`.
- `pass_through_client_auth` actually forwards the client's `Authorization` header.
- `n!=1` JSON responses are returned unmodified (no wrap of only the first choice).
- Request body size is enforced on the actual body; invalid JSON is 400 not 500.
- Docker image creates a writable `/data` for user `cradle`.
- L2 startup refuses `WEB_CONCURRENCY` / `UVICORN_WORKERS` other than 1.
- Upstream connection failures return OpenAI JSON 502 instead of an uncaught 500.
- The `cradle_l2_points` gauge is updated on each L2 write, not only every 300s by the purge loop, so a fresh instance no longer reports 0 L2 points while actively serving semantic hits. The purge loop still reconciles the exact count. (#4)
