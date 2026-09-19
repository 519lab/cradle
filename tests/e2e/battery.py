#!/usr/bin/env python3
"""End-to-end measurement battery for a *live* Cradle instance.

Unlike the fixture eval (``tests/eval``), which measures compression only on
chatty golden prompts, this drives realistic Open-WebUI-shaped traffic against a
running Cradle and reports the three numbers that actually decide the roadmap,
each on the path where it is meaningful:

  1. Cache hit rate      — from ``X-Cradle-Cache`` (HIT-L1/HIT-L2/MISS/BYPASS).
  2. Bypass rate         — a single bucket; the *reason* is inferred from the
                           request shape we sent (Cradle exposes no bypass-reason
                           header, verified against source), reported separately
                           via a labelled conformance block, never mixed into the
                           traffic-rate denominator.
  3. Compression savings — inbound vs upstream prompt tokens, measured ONLY on
                           non-stream fresh misses (``X-Cradle-Cache-Control:
                           no-store``). Streaming responses strip
                           ``X-Cradle-Upstream-Tokens`` (the count is unknown until
                           the body has streamed), so per-request compression is
                           not computable there.

This is a *diagnostic driver*, not a pass/fail test — it is deliberately kept out
of the pytest suite (no ``test_`` names, no assertions on absolute numbers). Point
it at a host and read the numbers. It requires a reachable upstream and will make
real (billable, if the upstream bills) completion calls.

Usage:
    uv run python tests/e2e/battery.py --base http://192.168.50.30:8000
    uv run python tests/e2e/battery.py --base http://192.168.50.25:8000 --json out.json

The preflight block asserts the instance is actually caching (features.cache on)
and reports the observed ``cache_tool_streams`` behaviour (#44) before any
scenario runs — because if caching is off, every response is BYPASS and every
number below is meaningless.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

DEFAULT_BASE = "http://192.168.50.30:8000"
DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
DEFAULT_KEY = "battery"  # tenant (forwarded Authorization is the cache tenant)

# A large shared system prompt — the #40 shape: the same instructions ride on
# every request, so the interesting compression/caching happens on the user turn.
BIG_SYS = (
    "You are a direct, precise technical assistant. Rules: Answer the user's request "
    "directly. Do NOT call, invoke, or reference any tool, function, or note-taking action "
    "(e.g. write_note, ask_user, calculate_timestamp) unless the user's message explicitly "
    "asks you to. For every prompt in this session, respond in plain text only. Do not narrate "
    "your reasoning. Follow all formatting instructions literally. Be concise. No preamble, no "
    "filler. If code is requested make it correct and runnable."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "save a note",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

# Open-WebUI internal task templates (fire constantly, NO tools) — the repeaters
# that the cache is meant to catch. <<C>> is a placeholder we substitute (str
# .replace, not str.format — the tags template contains literal JSON braces).
TASK_TITLE = (
    "### Task: Generate a concise title for the content. Respond with a single title only. "
    "No quotes, no explanation. ### Content: <<C>>"
)
TASK_TAGS = (
    "### Task: Generate 1-3 broad tags categorizing the main themes of the conversation. "
    '### Output: JSON only, format {"tags":[...]} ### Content: <<C>>'
)
TASK_FOLLOWUP = (
    "Generate 2-4 follow-up questions based on the content below. Actionable, concise. "
    "Content: <<C>>"
)

TASK_CONTENTS = [
    "A user asked about migrating a website to a new server.",
    "A discussion about Python data types and memory.",
    "How to bake sourdough bread at high altitude.",
]

# Real user chat turns (the dominant real shape). Used both with tools+stream
# (the #44 path) and plain (non-stream, no tools).
CHAT_TURNS = [
    "Explain how airplanes generate lift in simple terms.",
    "Write a short tactical briefing framing grocery shopping as a military operation.",
    "What's the difference between a process and a thread?",
    "Give me a one-paragraph summary of photosynthesis.",
    "How do I reverse a linked list in Python?",
]

# A multi-turn conversation that grows past a typical max_messages window.
MULTITURN = [
    ("user", "I'm building a CLI tool in Python. Where should config live?"),
    ("assistant", "Follow the XDG spec: ~/.config/<tool>/config.toml on Linux."),
    ("user", "How do I read TOML in the stdlib?"),
    ("assistant", "Python 3.11+ ships tomllib; open the file in binary mode."),
    ("user", "And to write it back out?"),
    ("assistant", "tomllib is read-only; use the third-party tomli-w to serialize."),
    ("user", "Summarize everything you've told me so far as a checklist."),
]


class Client:
    """Minimal blocking client. Returns (headers_lower, body_bytes, seconds, err)."""

    def __init__(self, base: str, model: str, key: str):
        self.base = base.rstrip("/")
        self.model = model
        self.key = key

    def post(self, body: dict, extra_headers: dict | None = None):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions", data=data, method="POST"
        )
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("Content-Type", "application/json")
        for k, v in (extra_headers or {}).items():
            req.add_header(k, v)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                h = {k.lower(): v for k, v in r.headers.items()}
                return h, r.read(), time.time() - t0, None
        except urllib.error.HTTPError as e:
            return {k.lower(): v for k, v in e.headers.items()}, e.read(), time.time() - t0, e.code
        except Exception as e:  # noqa: BLE001 — a live driver reports, never crashes mid-run
            return {}, b"", time.time() - t0, str(e)


def cache_outcome(h: dict) -> str:
    return h.get("x-cradle-cache", "?")


def tokens(h: dict) -> tuple[int, int]:
    """(inbound, upstream) prompt tokens. upstream is absent on streaming."""
    inb = int(h.get("x-cradle-inbound-tokens", 0) or 0)
    up_raw = h.get("x-cradle-upstream-tokens")
    up = int(up_raw) if up_raw not in (None, "") else -1  # -1 = not reported (stream)
    return inb, up


def task_body(model: str, tmpl: str, content: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": BIG_SYS},
            {"role": "user", "content": tmpl.replace("<<C>>", content)},
        ],
    }


# ---------------------------------------------------------------------------
# Preflight: prove the instance is caching, and observe the #44 flag behaviour.
# ---------------------------------------------------------------------------
def preflight(c: Client) -> dict:
    """Abort the run if caching is off; report the live cache_tool_streams state.

    Uses a unique nonce so the first probe is a guaranteed cold MISS regardless of
    what a prior battery run left in the cache — otherwise a warm cache reads as a
    HIT on the first request and the features.cache check falsely reports "off".
    """
    out: dict = {"base": c.base}
    nonce = f"preflight-{time.time_ns()}"
    probe = task_body(c.model, TASK_TITLE, nonce)

    h1, _, _, e1 = c.post(probe)
    h2, _, _, e2 = c.post(probe)  # identical → must hit on repeat if caching is on
    first, repeat = cache_outcome(h1), cache_outcome(h2)
    out["preflight_first"] = first
    out["preflight_repeat"] = repeat
    out["features_cache_on"] = (first == "MISS" and repeat in ("HIT-L1", "HIT-L2"))
    if e1 or e2:
        out["error"] = f"preflight request failed: {e1 or e2}"

    # #44 behavioural probe. A single request's label is NOT sufficient: MISS on
    # the first request only tells us the request wasn't refused up front, and
    # cache_tool_streams defaults to True in code — so a pre-#44 build with the
    # flag present but the replay path absent can still yield a first-request MISS.
    # The #44 signature is the REPEAT: a no-tool-call stream+tools response is only
    # replayed as HIT-L1 when the passthrough-cache path is actually live. Send an
    # identical pair (nonce-keyed, cold) and read the second label.
    tool_nonce = f"Preflight #44 probe {time.time_ns()}: reply in one short sentence, no tools."
    tool_body = {
        "model": c.model,
        "stream": True,
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": BIG_SYS},
            {"role": "user", "content": tool_nonce},
        ],
    }
    ht1, _, _, _ = c.post(tool_body)
    ht2, _, _, _ = c.post(tool_body)
    first_tool, repeat_tool = cache_outcome(ht1), cache_outcome(ht2)
    out["stream_tools_first"] = first_tool
    out["stream_tools_repeat"] = repeat_tool
    # Three-valued, because first-label alone conflates two live states:
    #   BYPASS               -> "off": flag off or path absent (is_cacheable False).
    #   MISS then a cache hit -> "live": #44 path present AND the stream caches.
    #   MISS then MISS       -> "live-not-caching": the path IS present (MISS, not
    #                           BYPASS, means is_cacheable returned True), but the
    #                           write gate refused the response — e.g. the stream
    #                           carried no parseable terminal finish_reason
    #                           (reason="finish_None"), or a tool call, or the
    #                           model's two answers differed. NOT "off".
    if first_tool == "BYPASS":
        state = "off"
    elif first_tool == "MISS" and repeat_tool in ("HIT-L1", "HIT-L2"):
        state = "live"
    elif first_tool == "MISS":
        state = "live-not-caching"
    else:
        state = f"inconclusive({first_tool}/{repeat_tool})"
    out["cache_tool_streams_state"] = state
    out["cache_tool_streams_on"] = state == "live"
    return out


# ---------------------------------------------------------------------------
# Traffic scenarios (populate the hit/bypass rate denominator). Each request
# is sent twice to measure the repeat-hit rate.
# ---------------------------------------------------------------------------
def run_traffic(c: Client) -> list[dict]:
    rows: list[dict] = []

    def record(scenario: str, expect_reason: str | None, h: dict, err) -> None:
        inb, up = tokens(h)
        rows.append(
            {
                "scenario": scenario,
                "cache": cache_outcome(h),
                "inbound": inb,
                "upstream": up,  # -1 when the header was stripped (streaming)
                "bypass_reason": expect_reason,  # shape-inferred, not from a header
                "error": err,
            }
        )

    # A) Open-WebUI task calls: no tools, non-stream — the cache's bread and butter.
    for tmpl in (TASK_TITLE, TASK_TAGS, TASK_FOLLOWUP):
        for content in TASK_CONTENTS:
            body = task_body(c.model, tmpl, content)
            for _ in range(2):
                h, _, _, err = c.post(body)
                record("task", None, h, err)

    # B) Real chat turns WITH tools, streaming — the #44 path. At default config
    #    this is a cacheable MISS then HIT-L1 (no tool call); it is *not* a bypass.
    for turn in CHAT_TURNS:
        body = {
            "model": c.model,
            "stream": True,
            "tools": TOOLS,
            "messages": [
                {"role": "system", "content": BIG_SYS},
                {"role": "user", "content": turn},
            ],
        }
        for _ in range(2):
            h, _, _, err = c.post(body)
            record("chat_tools_stream", None, h, err)

    # C) Same chat turns, non-stream + no tools — the plainly cacheable shape.
    for turn in CHAT_TURNS:
        body = {
            "model": c.model,
            "messages": [
                {"role": "system", "content": BIG_SYS},
                {"role": "user", "content": turn},
            ],
        }
        for _ in range(2):
            h, _, _, err = c.post(body)
            record("chat_plain", None, h, err)

    # D) A multi-turn conversation exceeding a typical window, sent twice.
    mt_msgs = [{"role": "system", "content": BIG_SYS}] + [
        {"role": role, "content": text} for role, text in MULTITURN
    ]
    for _ in range(2):
        h, _, _, err = c.post({"model": c.model, "messages": mt_msgs})
        record("multiturn", None, h, err)

    return rows


# ---------------------------------------------------------------------------
# Conformance block: one deliberate request per bypass reason, kept OUT of the
# traffic-rate denominator. Confirms each shape still bypasses (or, for
# stream+tools, does whatever cache_tool_streams dictates) post-#44.
# ---------------------------------------------------------------------------
def run_conformance(c: Client, tool_stream_state: str) -> list[dict]:
    rows: list[dict] = []

    def probe(reason: str, body: dict, expect: set[str]) -> None:
        h, _, _, err = c.post(body)
        got = cache_outcome(h)
        rows.append(
            {
                "reason": reason,
                "expected": "|".join(sorted(expect)),
                "got": got,
                "ok": (got in expect),
                "error": err,
            }
        )

    # A nonce keeps the request text distinct per run; it does not change the
    # classification (the bypass decision is on request shape, not content), but it
    # avoids surprises if these prompts collide with traffic. The expectations are
    # SETS: a bypass reason must always BYPASS regardless of cache warmth, while a
    # cacheable shape (stream+tools with the flag on) is valid as MISS *or* a
    # warm-cache HIT — both prove it was not bypassed.
    n = time.time_ns()
    base_msgs = [
        {"role": "system", "content": BIG_SYS},
        {"role": "user", "content": f"Say hello. (conf {n})"},
    ]
    CACHEABLE = {"MISS", "HIT-L1", "HIT-L2"}
    # n != 1 → BYPASS (never cacheable)
    probe("n!=1", {"model": c.model, "n": 2, "messages": base_msgs}, {"BYPASS"})
    # stream + logprobs → BYPASS
    probe(
        "stream+logprobs",
        {"model": c.model, "stream": True, "logprobs": True, "messages": base_msgs},
        {"BYPASS"},
    )
    # non-text multimodal content part → BYPASS (uncacheable_reason="non_text_part")
    nontext_msgs = [
        {"role": "system", "content": BIG_SYS},
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Describe this. (conf {n})"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}],
        },
    ]
    probe("non_text_part", {"model": c.model, "messages": nontext_msgs}, {"BYPASS"})
    # stream + tools → BYPASS only when the flag is off. When the #44 path is
    # present ("live" or "live-not-caching") the request is cacheable, so a MISS is
    # correct even if the write is later refused; a warm "live" host may return HIT.
    stream_tools_expect = {"BYPASS"} if tool_stream_state == "off" else CACHEABLE
    probe(
        "stream+tools",
        {"model": c.model, "stream": True, "tools": TOOLS, "messages": base_msgs},
        stream_tools_expect,
    )
    return rows


# ---------------------------------------------------------------------------
# Compression measurement: non-stream fresh misses only, forced with no-store so
# nothing is served from cache and upstream tokens are always reported.
# ---------------------------------------------------------------------------
def run_compression(c: Client) -> list[dict]:
    rows: list[dict] = []
    # no-store AND no-cache: no-store alone only prevents the WRITE — it does not
    # skip the cache READ, so a turn already cached by run_traffic() would return an
    # L1 HIT with upstream_tokens=0 and read as a false "+100% savings". no-cache
    # forces a fresh upstream call; a per-turn nonce guarantees no collision even
    # on a warm cache. This is the only path where upstream tokens are reported
    # (streaming strips the header), so it is the only place compression is real.
    headers = {"X-Cradle-Cache-Control": "no-store, no-cache"}
    nonce = time.time_ns()
    for turn in CHAT_TURNS:
        body = {
            "model": c.model,
            "messages": [
                {"role": "system", "content": BIG_SYS},
                {"role": "user", "content": f"{turn} (ref {nonce})"},
            ],
        }
        h, _, _, err = c.post(body, extra_headers=headers)
        inb, up = tokens(h)
        cache = cache_outcome(h)
        # A real compression number requires a MISS (upstream actually called).
        # Surface anything else rather than averaging a bogus 0 into the total.
        rows.append(
            {"turn": turn[:40], "inbound": inb, "upstream": up, "cache": cache, "error": err}
        )
    return rows


def report(pre: dict, traffic: list[dict], conf: list[dict], comp: list[dict]) -> dict:
    line = "=" * 72
    print(f"\n{line}\nCRADLE E2E BATTERY — {pre['base']}\n{line}")

    print("\n-- PREFLIGHT --")
    print(f"  features.cache on:        {pre.get('features_cache_on')} "
          f"(first={pre.get('preflight_first')}, repeat={pre.get('preflight_repeat')})")
    print(f"  cache_tool_streams (#44): {pre.get('cache_tool_streams_state')} "
          f"(stream+tools first={pre.get('stream_tools_first')}, "
          f"repeat={pre.get('stream_tools_repeat')})")
    if pre.get("cache_tool_streams_state") == "live-not-caching":
        print("    NOTE: the #44 path IS present (MISS, not BYPASS), but tool-stream "
              "responses are not being cached — check cradle_cache_write_skips_total "
              "(likely finish_None: the upstream stream carries no parseable terminal "
              "finish_reason, so the fail-closed write gate refuses it).")
    if not pre.get("features_cache_on"):
        print("\n  ABORT: caching is OFF on this instance — every response is BYPASS "
              "and all numbers below are meaningless. Fix features.cache and re-run.")
        return {"preflight": pre, "aborted": True}

    # Hypothesis (state before reading): #44 makes the chat_tools_stream REPEAT a
    # HIT-L1 rather than a BYPASS. Report whether it won.
    print("\n-- HYPOTHESIS (pre-registered) --")
    print("  #44 moves chat_tools_stream from BYPASS to a cacheable MISS→HIT-L1 pair.")

    total = len(traffic)
    outcomes: dict[str, int] = defaultdict(int)
    by_scn: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors = 0
    for r in traffic:
        outcomes[r["cache"]] += 1
        by_scn[r["scenario"]][r["cache"]] += 1
        if r["error"]:
            errors += 1

    print("\n-- TRAFFIC MIX (each request sent twice) --")
    for scn, cc in by_scn.items():
        print(f"  {scn:20s} {dict(cc)}")
    hits = outcomes.get("HIT-L1", 0) + outcomes.get("HIT-L2", 0)
    byp = outcomes.get("BYPASS", 0)
    print(f"\n  overall: {dict(outcomes)}")
    print(f"  hit rate:    {hits}/{total} = {hits / total * 100:.0f}%" if total else "  no traffic")
    print(f"  bypass rate: {byp}/{total} = {byp / total * 100:.0f}%" if total else "")
    print(f"  errors:      {errors}")

    ts = by_scn.get("chat_tools_stream", {})
    hyp_won = ts.get("HIT-L1", 0) > 0 and ts.get("BYPASS", 0) == 0
    print(f"\n  HYPOTHESIS {'WON' if hyp_won else 'LOST'}: "
          f"chat_tools_stream outcomes = {dict(ts)} "
          f"(expected some HIT-L1, zero BYPASS)")
    if not hyp_won:
        if ts.get("BYPASS", 0) > 0:
            print("    -> BYPASS present: the #44 path is NOT live on this instance "
                  "(pre-#44 build, or cache_tool_streams=false).")
        elif ts.get("HIT-L1", 0) == 0 and ts.get("MISS", 0) > 0:
            print("    -> all MISS, no HIT-L1: #44 path IS live but these responses "
                  "carried tool calls (correctly suppressed from caching), OR the "
                  "model's answers varied between the two identical sends. This is "
                  "NOT a regression — rule out the tool-call case before concluding.")

    print("\n-- BYPASS CONFORMANCE (out of the rate above) --")
    for r in conf:
        mark = "ok" if r["ok"] else "MISMATCH"
        print(f"  {r['reason']:18s} expected={r['expected']:7s} got={r['got']:7s} [{mark}]")

    print("\n-- COMPRESSION (non-stream, no-store+no-cache fresh misses only) --")
    ti = tu = 0
    skipped = 0
    for r in comp:
        i, u, cache = r["inbound"], r["upstream"], r.get("cache", "?")
        # Only a genuine MISS with a reported upstream count is a real datapoint.
        # A HIT (upstream=0) or a stripped header (-1) would fake a +100% saving.
        if cache != "MISS" or u < 0:
            skipped += 1
            print(f"  {r['turn']:42s} SKIPPED (cache={cache}, upstream={u}) — not a fresh miss")
            continue
        ti += i
        tu += u
        sav = (i - u) / i * 100 if i else 0.0
        print(f"  {r['turn']:42s} inbound={i:5d} upstream={u:5d} savings={sav:+5.1f}%")
    if ti:
        print(f"  {'TOTAL':42s} inbound={ti:5d} upstream={tu:5d} "
              f"savings={(ti - tu) / ti * 100:+5.1f}%")
        print("  (negative ⇒ compression ADDS tokens for this traffic; real savings come from caching)")
    if skipped:
        print(f"  ({skipped} turn(s) skipped — no valid fresh-miss measurement)")

    return {
        "preflight": pre,
        "traffic_outcomes": dict(outcomes),
        "hit_rate": hits / total if total else None,
        "bypass_rate": byp / total if total else None,
        "errors": errors,
        "hypothesis_won": hyp_won,
        "conformance": conf,
        "compression_total": {"inbound": ti, "upstream": tu, "valid_turns": len(comp) - skipped},
        "compression_savings_pct": ((ti - tu) / ti * 100) if ti else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Cradle e2e measurement battery.")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Cradle base URL")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key", default=DEFAULT_KEY, help="tenant / Authorization bearer")
    ap.add_argument("--json", dest="json_out", help="write the summary dict as JSON here")
    args = ap.parse_args()

    c = Client(args.base, args.model, args.key)
    print(f"Battery → {args.base} (model={args.model}) ...")

    pre = preflight(c)
    if not pre.get("features_cache_on"):
        summary = report(pre, [], [], [])
    else:
        traffic = run_traffic(c)
        conf = run_conformance(c, pre.get("cache_tool_streams_state", "off"))
        comp = run_compression(c)
        summary = report(pre, traffic, conf, comp)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0 if not summary.get("aborted") else 1


if __name__ == "__main__":
    sys.exit(main())
