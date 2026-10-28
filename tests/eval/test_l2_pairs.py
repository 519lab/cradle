from __future__ import annotations

import json
from pathlib import Path

import pytest

from cradle.embeddings.fastembed import FastEmbedEmbedder

pytestmark = pytest.mark.embed

PAIRS = Path(__file__).parent / "l2_pairs.jsonl"


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_bge_threshold(tmp_path) -> None:
    emb = FastEmbedEmbedder(cache_dir=tmp_path / "fe")
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    hits = [r for r in rows if r["expect"] == "hit"]
    misses = [r for r in rows if r["expect"] == "miss"]
    assert len(hits) >= 10 and len(misses) >= 10
    for r in hits:
        score = _cos(emb.embed(r["a"]), emb.embed(r["b"]))
        assert score >= 0.90, (r["id"], score)
    for r in misses:
        if r.get("same_user") is False or "seed_a" in r:
            continue
        score = _cos(emb.embed(r["a"]), emb.embed(r["b"]))
        assert score < 0.90, (r["id"], score)
