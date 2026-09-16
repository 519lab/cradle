# Changelog

## [Unreleased]

### Added

- v1 Cradle pass-through API gateway: OpenAI-compatible `/v1/chat/completions` with L1 diskcache, L2 Qdrant local + FastEmbed, rule-based compression, prefix/suffix reconstruction, Prometheus metrics, and a fixture eval harness. No live LLM required for tests.

### Changed

- Product copy: Cradle is a gateway clients point at, not a local companion app for a model runtime.
