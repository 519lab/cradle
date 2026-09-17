# Changelog

## [Unreleased]

### Added

- L2 semantic-cache **cross-encoder rerank** (default on, `features.l2_rerank`): after the cheap numbers/negation guard, a `BAAI/bge-reranker-base` cross-encoder re-scores the surviving candidate and rejects it below a threshold. This closes the **named-entity-swap** gap the guard cannot see — verified live, "capital of Germany" now returns Berlin instead of the cached "Paris" (reject score −3.5). Runs only on candidates that already passed the cosine gate and guard; **fails open** (an unavailable or erroring reranker serves the already-twice-gated hit rather than forcing a miss). Observable via `X-Cradle-Rerank` (`pass:<score>` / `reject:<score>` / `fail-open`) and `cradle_l2_rerank_rejects_total` / `cradle_l2_rerank_fail_open_total`. **Residual gap:** antonym swaps (closest/farthest, largest/smallest-direction) score in the paraphrase range and are not reliably caught — same fundamental limitation as negation. Cost: the reranker adds ~15–40ms per L2 hit and ~1.1GB to the image — see the DESIGN decision on the L2 latency budget. (#5)
- L2 semantic-cache **precision guard**: before serving a semantic hit, Cradle now compares the query against the stored candidate and rejects the hit when their numbers or negation differ, then falls through to a real upstream call. This stops the cache from returning a stale wrong answer to a near-miss prompt — e.g. a "convert 200 USD" request no longer replays the "100 USD" answer, and "is X true?" no longer replays the answer to "is X NOT true?". A rejection is observable via the new `X-Cradle-Guard: reject:<reason>` response header and the `cradle_l2_guard_rejects_total{reason}` metric. Known gap: single-entity swaps (capital of France vs Germany) are not caught by this guard and need a cross-encoder (tracked in #5). (#5)
- v1 Cradle pass-through API gateway: OpenAI-compatible `/v1/chat/completions` with L1 diskcache, L2 Qdrant local + FastEmbed, rule-based compression, prefix/suffix reconstruction, Prometheus metrics, and a fixture eval harness. No live LLM required for tests.
- Runnable multi-stage Docker image (`uv sync --frozen`) and Compose stack. FastEmbed weights bake into `/opt/cradle/models/fastembed` so the `/data` volume does not hide them. Compose reaches a host-side upstream via `host.docker.internal`. CI builds `compose.ci.yml` and smokes `/healthz` + `/readyz`.
- Docker image now also bakes the `bge-reranker-base` cross-encoder (`BAKE_RERANK=1`, default) into `/opt/cradle/models/fastembed`, so a fresh container is self-contained and does not download the ~1.1GB reranker at startup warm-up (needing HF network access). Set `BAKE_RERANK=0` to skip when rerank is disabled.
- `HF_TOKEN` build arg (Dockerfile) + Compose passthrough, so the image bake steps can authenticate Hugging Face model downloads (gated or rate-limited pulls). Documented in `.env.example`; optional, empty = anonymous.

### Changed

- Renamed `compose.yml` to `docker-compose.yml`.
- `config/cradle.yaml` is now gitignored with `config/cradle.yaml.example` as the tracked template — copy the example to the real filename before running, so local config edits no longer conflict on every `git pull`.
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
