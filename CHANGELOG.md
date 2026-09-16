# Changelog

## [Unreleased]

### Added

- v1 Cradle pass-through API gateway: OpenAI-compatible `/v1/chat/completions` with L1 diskcache, L2 Qdrant local + FastEmbed, rule-based compression, prefix/suffix reconstruction, Prometheus metrics, and a fixture eval harness. No live LLM required for tests.

### Changed

- Product copy: Cradle is a gateway clients point at, not a local companion app for a model runtime.

### Fixed

- Bypass streams (`stream+tools`, `n!=1`, logprobs) now forward upstream SSE bytes instead of stripping everything but `delta.content`.
- Upstream stream HTTP errors return JSON OpenAI errors before SSE starts.
- Truncated streams no longer fake `finish_reason=stop`.
- `pass_through_client_auth` actually forwards the client's `Authorization` header.
- `n!=1` JSON responses are returned unmodified (no wrap of only the first choice).
- Request body size is enforced on the actual body; invalid JSON is 400 not 500.
- Docker image creates a writable `/data` for user `cradle`.
- L2 startup refuses `WEB_CONCURRENCY` / `UVICORN_WORKERS` other than 1.
