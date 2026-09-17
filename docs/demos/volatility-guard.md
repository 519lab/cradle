# DEMO — innovation #4: volatility guard

Branch: `innovation/volatility-guard` (from `develop`).

## What it is

Prompts that ask about something time-sensitive are stored with a short TTL
(`cache.volatile_ttl_s`, default 300 s) instead of the 24 h default. Reasons:
`time` (latest/current/today/as of/what time is it…), `market` (price, exchange
rate, "how much does … cost", crypto), `weather`, `news`. Classification runs on
user turns only, so a system prompt that carries today's date does not trip it.
An explicit client `X-Cradle-Cache-TTL` always wins. `volatile_ttl_s: 0` means
"never store volatile prompts".

Side fix included: L2 → L1 promotion now honors the per-request TTL (it used to
always write the default 24 h, even for a client-supplied short TTL).

## How to run

```bash
uv run pytest tests/test_volatility.py        # 24 tests: every pattern pinned + pipeline
curl -si http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"what is the latest version of python"}]}' \
  | grep -i x-cradle-volatile
# x-cradle-volatile: time
```

## What was verified

- Stored record for a `time` prompt has `ttl_s == 300`; a stable prompt keeps
  `86400`; `X-Cradle-Cache-TTL: 3600` beats the guard; `volatile_ttl_s: 0`
  stores nothing (repeat is a `MISS`); `volatility_guard: false` restores the
  default; 12 volatile phrasings and 6 look-alike stable ones pinned.
- Live smoke (real bge-small + bge-reranker-base, fake upstream), stored L1
  records dumped from diskcache afterwards:

  | Prompt | Headers | Stored `ttl_s` |
  |---|---|---|
  | what is the latest version of python | `MISS`, `X-Cradle-Volatile: time` | 300 |
  | what is the capital of France | `MISS` (no volatile header) | 86400 |
  | what is the weather in Toronto | `MISS`, `X-Cradle-Volatile: weather` | 300 |
  | …weather in Toronto right now + `X-Cradle-Cache-TTL: 3600` | `HIT-L2` (client TTL wins, no volatile header) | 3600 |
  | which is the latest python version | `HIT-L2` cosine 0.9879, `X-Cradle-Volatile: time` | 300 (promoted with the short TTL) |

  `/metrics`: `cradle_volatile_prompts_total{reason="time"} 2`, `{reason="weather"} 1`.

## What is stubbed

Nothing. The pattern table is deliberately small and conservative; false
positives only shorten a TTL and are visible in the header and metric.

## Next increment

Let the audit log from innovation #1 grade volatile hits separately, so the
table can be tuned on evidence: a `market` prompt whose fresh answer disagrees
with the cached one within 300 s argues for a shorter `volatile_ttl_s`.
