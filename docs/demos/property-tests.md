# DEMO — innovation #6: property-based tests

Branch: `innovation/property-tests` (from `develop`).

## What it is

`tests/test_properties.py` encodes the invariants the example tests only
sample, and lets Hypothesis search for counterexamples:

| Invariant | Module |
|---|---|
| `restore(extract(t)) == t` for any text; spans are disjoint, ordered, and slice back to their text | `compress/guards.py` |
| every protected span survives `strip_fluff` | `compress/guards.py` + `rules.py` |
| `strip_fluff` is idempotent and never grows text | `compress/rules.py` |
| canonical text ignores whitespace outside protected spans | `normalize.py` |
| `l1_key` ignores field order and `tools: []` vs `null`, changes on any extra | `normalize.py` |
| `sampling_fingerprint` ignores messages, changes with temperature | `normalize.py` |
| `synthesize_sse` → `parse_and_accumulate` reproduces the stored body, finish reason, `[DONE]`, and usage gating | `gateway/sse.py` |

Default 200 examples per property (fast in CI); `CRADLE_HYPOTHESIS_MAX=5000`
for a deeper local run. `hypothesis` is a `dev` dependency only.

## What it found (and what was done)

1. **`strip_fluff` was not idempotent** — three shrinking counterexamples in
   sequence: `" please a"` (leading space defeats the `^`-anchored rule),
   `"{please \n,"` (a newline around the closing pleasantry defeats the
   `$`-anchored one, which only skips `[ ,.!]`), and `"just please a"`
   (removing the filler exposes a pleasantry the leading rule already ran
   past). **Fixed** in `compress/rules.py`: the rule pass now iterates to a
   fixed point, which the never-grows property guarantees terminates. 3000
   examples pass. Cache keys are unaffected: they hash canonical, not
   compressed, text. No `pipeline_version` bump recommended: the rules did not
   change, they now also apply to inputs they were always meant to cover.
2. **`_canonicalize_text` is not a fixed point** — `"a\n```\n"` →
   `"a ```\n"` → `"a ```"`: whitespace collapse joins the line before an
   unclosed fence onto the opener, so on a second pass the fence is no longer
   at a line start. Nothing re-canonicalizes canonical text, so this is
   **pinned as a strict xfail** with the counterexample rather than fixed;
   fixing it would change L1 keys for fenced prompts and needs a
   `HASH_SCHEMA_VERSION` bump.

## How to run

```bash
uv run pytest tests/test_properties.py                        # 9 passed, 1 xfailed
CRADLE_HYPOTHESIS_MAX=5000 uv run pytest tests/test_properties.py
```

## Next increment

A stateful Hypothesis `RuleBasedStateMachine` over the pipeline (miss / hit /
refresh / no-store / TTL sequences against a fake upstream) to check the cache
never serves an answer the upstream did not produce for an equal canonical key.
