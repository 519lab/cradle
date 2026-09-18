"""Unit tests for the L2 rerank stage (issue #5 entity gap).

Uses a fake reranker so no model is loaded. The real BAAI/bge-reranker-base
separation is validated empirically (see research notes), not in unit tests.
"""

from __future__ import annotations

import pytest

from cradle.cache.rerank import passes


class FakeReranker:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, query: str, candidate: str) -> float:
        return self._score


class BoomReranker:
    def score(self, query: str, candidate: str) -> float:
        raise RuntimeError("onnx exploded")


def test_score_above_threshold_passes() -> None:
    ok, score = passes(FakeReranker(5.0), "q", "c", threshold=4.0)
    assert ok is True
    assert score == 5.0


def test_score_below_threshold_rejected() -> None:
    ok, score = passes(FakeReranker(3.4), "q", "c", threshold=4.0)
    assert ok is False
    assert score == 3.4


def test_score_at_threshold_passes() -> None:
    ok, score = passes(FakeReranker(4.0), "q", "c", threshold=4.0)
    assert ok is True
    assert score == 4.0


def test_missing_reranker_fails_open() -> None:
    ok, score = passes(None, "q", "c", threshold=4.0)
    assert ok is True
    assert score is None


def test_model_error_fails_open() -> None:
    """A broken reranker serves the candidate (it already passed cosine+guard)."""
    ok, score = passes(BoomReranker(), "q", "c", threshold=4.0)
    assert ok is True
    assert score is None


@pytest.mark.parametrize(
    "score,expected",
    [(-4.68, False), (3.49, False), (4.81, True), (9.53, True)],
)
def test_calibration_points(score: float, expected: bool) -> None:
    """Representative scores from the widened fixture at the default ~4.0 line."""
    ok, _ = passes(FakeReranker(score), "q", "c", threshold=4.0)
    assert ok is expected


def test_cuda_without_provider_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """cuda=True on a build with no CUDAExecutionProvider fails with a clear
    message pointing at the GPU image, not the raw fastembed ValueError (#31)."""
    import onnxruntime as ort

    from cradle.cache.rerank import FastEmbedReranker

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="rerank_device"):
        FastEmbedReranker("BAAI/bge-reranker-base", cuda=True)
