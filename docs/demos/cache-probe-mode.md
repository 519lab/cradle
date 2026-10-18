# DEMO — innovation #3: cache probe mode

Branch: `innovation/cache-probe-mode` (from `develop`).

## What it is

`X-Cradle-Cache-Control: probe` runs the full read-side decision (L1 lookup,
embed, L2 top-K, numbers/negation guard, cross-encoder rerank) and returns a
`cradle.probe` JSON explanation instead of a completion. It never writes to L1
or L2 (no promote, no writeback) and never calls upstream.

## How to run

```bash
uv run pytest tests/test_probe.py            # 5 tests, no models needed
uv run python -m cradle                      # then:
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -H "X-Cradle-Cache-Control: probe" \
  -d '{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"Which city is the capital of France?"}]}'
```

## What was verified live (real bge-small + bge-reranker-base, fake upstream)

1. Seed `What is the capital of France?` → `MISS`, upstream called once.
2. Probe `Which city is the capital of France?` → `HIT-L2`, one candidate
   `cosine 0.973013 · guard pass · rerank pass:8.3960 · served true`,
   `would_call_upstream: false`. Upstream not called.
3. Probe `What is the capital of Germany?` → `MISS`, zero candidates above the
   0.90 cosine floor, `would_call_upstream: true`. Upstream not called.
4. Real request for the Germany prompt afterwards → `MISS` (the probe wrote
   nothing), answered `Berlin.` from upstream.
5. `/metrics`: `cradle_cache_probes_total{cache="l2"} 1`, `{cache="miss"} 1`;
   `cradle_cache_misses_total 2` (probes are not counted as hits or misses).

## What works / what is stubbed

- Works: JSON explanation for BYPASS, HIT-L1, HIT-L2 (with every examined
  candidate best-first, including guard- and rerank-rejected ones), and MISS.
  Normal `X-Cradle-*` headers are set too, plus `X-Cradle-Probe: 1`.
- Nothing stubbed.
- Not in scope: probe for streaming requests returns the same JSON (stream flag
  is not hashed, so the decision is identical); the response is never SSE.

## Next increment

Feed probe output into a threshold-calibration script
(`python -m cradle.eval.calibrate pairs.jsonl`) that reports cosine/rerank
separation between should-hit and should-miss pairs — the labeled pairs come
from innovation #1's audit log.
