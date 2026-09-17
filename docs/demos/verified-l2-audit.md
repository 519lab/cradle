# DEMO — innovation #1: verified L2 (audit sampling, learned floors, self-heal)

Branch: `innovation/verified-l2-audit` (from `develop`).

## What it is

`l2.audit_rate` (default `0`, off) samples served L2 hits. For each sampled
hit, a background task re-asks upstream with the same request a miss would
send and a judge decides whether the fresh answer agrees with the served one.

- `cradle_l2_audit_total{verdict}` — a measured false-hit rate on real traffic.
- `{data_dir}/audits.jsonl` — one labeled row per audit (`query_similarity`,
  `guard`, `rerank`, `judge`, `answer_score`, `verdict`, keys). Prompt text is
  included only with `l2.audit_log_text: true`.
- **Per-entry floor.** An entry judged wrong at similarity `s` stores
  `audit_floor = s` and refuses future matches at `≤ s`
  (`X-Cradle-Guard: reject:audit-floor`). Monotone, no cold start.
- **Self-heal.** On disagree the fresh answer is written under the querying
  prompt's own key (L1 + L2), so the next paraphrase hits the right entry.
- The served response is never delayed; sampled hits carry
  `X-Cradle-Audit: scheduled`. Tasks are tracked on `Runtime.audit_tasks` and
  drained at shutdown.

## The judge — measured, and it changed the design

The first cut judged with answer-embedding cosine. A live run on the
documented antonym gap (closest vs farthest planet) exposed it: the wrong hit
was served, the audit ran, and cosine scored the two contradictory answers at
**0.871 → "agree"**. Measuring both candidates on the shipped bge models:

| pair | cosine | cross-encoder |
|---|---|---|
| Mercury closest / Mercury nearest (same) | 0.983 | 10.31 |
| "Paris." / "Paris is the capital of France." (same) | 0.828 | 7.18 |
| 100 USD ≈ 92 EUR, two phrasings (same) | 0.944 | 9.62 |
| Mercury closest / Neptune farthest (**wrong**) | 0.871 | −0.46 |
| "Paris." / "Berlin." (**wrong**) | 0.840 | −0.40 |
| dynamically typed / statically typed (**wrong**) | 0.921 | 2.10 |
| 100 USD / 200 USD (**wrong**) | 0.838 | −4.22 |
| "Yes, it is safe" / "No, it is not safe" (**wrong**) | 0.823 | −2.71 |

Cosine overlaps completely (same 0.83–0.98 vs wrong 0.73–0.92). The
cross-encoder Cradle already loads separates them with a gap from 2.1 to 7.2,
and the existing `rerank_threshold` of 4.0 sits in it. So `l2.audit_judge:
auto` uses the reranker when loaded; cosine is the fallback with a high
threshold (0.90) that prefers a false "disagree" (a miss) over a missed error.

## Verified live (real bge-small + bge-reranker-base, fake upstream)

1. Seed `Which planet is closest to the Sun?` → `MISS` → "Mercury is the closest…".
2. `Which planet is farthest from the Sun?` → **`HIT-L2`** cosine 0.911,
   reranker `pass:5.0785`, `X-Cradle-Audit: scheduled` → served "Mercury…"
   (the documented antonym gap, reproduced live).
3. Audit: judge `rerank`, `answer_score 0.526` → **`disagree`**; seed entry
   `audit_floor = 0.910920`; fresh "Neptune is the farthest…" written back
   under the query's key.
4. Same prompt again → `HIT-L1` → **"Neptune is the farthest planet from the Sun."**
5. `Which planet is nearest to the Sun?` → `HIT-L2` cosine 0.971 (above the
   floor, so the seed still serves) → audit `agree` at 10.31.
6. `What is the closest planet to the Sun?` → `HIT-L2` 0.989 → `agree` 10.31.
7. `/metrics`: `cradle_l2_audit_total{verdict="disagree"} 1`, `{verdict="agree"} 2`.

## How to run

```bash
uv run pytest tests/test_audit.py            # 12 tests, no models needed
CRADLE_L2__AUDIT_RATE=0.05 uv run python -m cradle
tail -f data/audits.jsonl
```

## What is stubbed / not in scope

- Nothing stubbed. An LLM-as-judge slots in behind `judge_answers` later.
- Audits reuse the client's forwarded `Authorization` after its response has
  completed — documented in README/CLAUDE.md; keep `audit_rate` at 1–5 %.
- The floor is per served entry only; it does not yet generalise across
  entries (vCache's per-entry model is the same scope).
- This branch does not include the `refresh` L2 fix (`fix/refresh-reembeds-for-l2`);
  self-heal does not depend on it because the hit path already has the query
  vector.

## Next increment

`python -m cradle.eval.calibrate data/audits.jsonl`: read the labeled rows and
report the cosine / rerank separation between agree and disagree, i.e. the
evidence to move `cosine_threshold` and `rerank_threshold` off their
hand-calibrated defaults. Then feed the NLI stage (INNOVATIONS #5) into the
same judge seam.
