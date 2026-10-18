# Cradle

Cradle is a caching proxy for OpenAI-compatible LLM APIs. Point your client at it instead of the provider; it answers repeated and near-repeated prompts from a local cache and forwards the rest upstream unchanged. Same `/v1/chat/completions` contract, same `Authorization` header.

Three layers handle each request:

1. **L1** — exact-match cache keyed on a SHA-256 of the generation-affecting fields (diskcache, sub-millisecond lookups).
2. **L2** — semantic cache: the prompt is embedded locally (FastEmbed `BAAI/bge-small-en-v1.5`) and matched against past prompts in an in-process Qdrant store.
3. **Miss** — Cradle strips conversational filler, calls upstream, wraps the answer, and writes both caches so the next similar prompt is a hit.

Everything on the cache path runs on CPU inside the gateway process. No Redis, no Qdrant server, no Ollama, no cloud embedding API. Cradle does not run a model of its own — upstream is whatever OpenAI-compatible API you configure.

## Quick start

```bash
cp .env.example .env                               # set CRADLE_UPSTREAM_BASE_URL if upstream isn't localhost:8080
cp config/cradle.yaml.example config/cradle.yaml   # your working config (the real file is gitignored)
uv sync --group dev
uv run python -m cradle
```

Cradle listens on `http://127.0.0.1:8000`. The default upstream is `http://127.0.0.1:8080/v1`; set `CRADLE_UPSTREAM_BASE_URL` to your provider (e.g. `https://api.openai.com/v1`).

Send it a request the way you'd send one to the provider:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

Cradle forwards your `Authorization` header upstream and uses it as the cache tenant, so entries never leak between keys. There's no Cradle-issued key by default; set `auth.keys` in the YAML if you want an allowlist instead.

## Docker

```bash
cp .env.example .env
cp config/cradle.yaml.example config/cradle.yaml
cp docker-compose-cpu.yml docker-compose.yml        # or docker-compose-gpu.yml; docker-compose.yml is gitignored
docker compose up --build
```

Two compose templates ship with the repo: `docker-compose-cpu.yml` (the default) and `docker-compose-gpu.yml` (CUDA rerank). Copy the one you want to `docker-compose.yml` and plain `docker compose up` finds it.

Both templates bind-mount `config/cradle.yaml` read-only into the container (`./config:/app/config:ro`) rather than baking it in, so a config change takes effect on `docker compose restart cradle` — no rebuild. With no host config file, Cradle runs on its built-in defaults. Cache state lives in the `cradle-data` volume at `/data`; the FastEmbed weights are baked into the image so the volume doesn't hide them.

The gateway is at `http://127.0.0.1:8000`. The default upstream under Compose is `http://host.docker.internal:8080/v1` (a server on the host); override it with `CRADLE_UPSTREAM_BASE_URL`.

## Pointing clients at Cradle

| Client | Where to set the base URL |
|---|---|
| OpenAI SDK / LiteLLM / llama.cpp | `OPENAI_BASE_URL=http://<cradle>:8000/v1` |
| Codex CLI | `openai_base_url`, or `[model_providers.cradle] base_url`, in `~/.codex/config.toml` |
| Grok CLI | `model.<id>.base_url` or `GROK_XAI_API_BASE_URL` |

Cradle routes each request to a backend by matching the request `model` against `routes` in `config/cradle.yaml` — for example `gpt-*` to OpenAI, `grok-*` to xAI, `*` to a local llama.cpp or LiteLLM. Models that match nothing fall back to `upstream.base_url`.

**Claude Code won't work against Cradle yet.** It speaks the Anthropic Messages API (`/v1/messages`), and Cradle only implements OpenAI's `/v1/chat/completions`. Point Claude Code at a proxy that already speaks Anthropic (LiteLLM, for instance), not at Cradle.

## Per-request cache controls

Send these headers to override the default behavior for a single request:

| Header | Effect |
|---|---|
| `X-Cradle-Cache-Control: no-store` | Answer normally, but don't store the response. |
| `X-Cradle-Cache-Control: no-cache` (or `refresh`) | Skip the cache, call upstream, store the fresh answer. |
| `X-Cradle-Cache-TTL: <seconds>` | Per-entry TTL, clamped to `cache.ttl_s`; `0` means don't store. |
| `X-Cradle-Cache-Control: probe` | Dry run — explain the L1/L2/guard/rerank decision without answering, writing, or calling upstream. |

`probe` is useful for tuning thresholds without spending a token. It returns a `cradle.probe` object listing every candidate the pipeline considered, best-first, so you can see exactly why a near-miss was rejected:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -H "X-Cradle-Cache-Control: probe" \
  -d '{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"Which city is the capital of France?"}]}'
```

## Verified L2 (audit sampling)

A semantic hit that's subtly wrong still returns a `200`, so you can't tell a good threshold from a bad one by watching hit rates alone. Set `l2.audit_rate` (e.g. `0.02`) and Cradle re-asks upstream, in the background, for that fraction of served L2 hits and compares the fresh answer to the one it served. Each audit:

- records the verdict in `cradle_l2_audit_total{verdict}` — a measured false-hit rate on your real traffic;
- raises a similarity floor on any entry judged wrong, so it won't be served on that near-miss again;
- caches the fresh answer under the querying prompt's own key;
- appends a row to `data/audits.jsonl` you can use to calibrate `cosine_threshold` and `rerank_threshold`.

Audits are real upstream calls made with the client's forwarded credentials after its response has already gone out, so keep the rate small. This is off by default.

## Metrics

Token savings from caching and compression:

```
(inbound_prompt_tokens - upstream_prompt_tokens) / inbound_prompt_tokens * 100
```

Prometheus exports `cradle_inbound_prompt_tokens_total`, `cradle_upstream_prompt_tokens_total`, `cradle_cache_hits_total{layer}`, `cradle_cache_misses_total`, and `cradle_latency_seconds{stage}`. `/metrics` requires the same Bearer key as the API. Run the offline eval with `uv run pytest tests/eval`.

## Operational notes

- **GPU rerank** (`l2.rerank_device: cuda`) cuts rerank latency substantially but needs the GPU image: `cp docker-compose-gpu.yml docker-compose.yml && docker compose up --build` on a host with an NVIDIA GPU and the Container Toolkit. Asking for `cuda` on the CPU image fails fast with a clear error. See `RUNBOOK.md` §1.4.
- **L2 scale.** The in-process Qdrant store isn't meant for more than ~20,000 points; past that, the path forward is a Qdrant server, not more local-mode tuning (server mode isn't in v1). Watch `cradle_l2_points`.
- **Volatility guard** (`cache.volatility_guard`, on by default). Prompts about time-sensitive things — "latest version", "current price", "today", weather, news — get a short TTL (`cache.volatile_ttl_s`, default 300s) instead of the usual 24h, so a correct-but-stale answer isn't replayed all day. An explicit `X-Cradle-Cache-TTL` always wins.

## More

- `RUNBOOK.md` — running Cradle in production.
- `DESIGN.md` — architecture and design decisions.
- `LICENSE` — MIT. Source at `519lab/cradle`.
