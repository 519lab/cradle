"""Cross-encoder rerank stage for the L2 semantic cache (issue #5, entity gap).

The numbers/negation guard (``guard.py``) catches near-misses that differ by a
number or a negation, but it is blind to single-entity swaps: "capital of
France" vs "Germany", "closest" vs "farthest planet". The bi-encoder that feeds
L2 is blind to them too (both score cosine >= 0.90).

A cross-encoder reads the query and the candidate *together* and scores true
relevance, which separates entity swaps that a bi-encoder cannot. Empirically,
on ``BAAI/bge-reranker-base`` (CPU ONNX via FastEmbed, already a dependency),
once the cheap guard has removed the number/negation classes the reranker is
itself blind to, genuine paraphrases score >= 4.8 and entity swaps score <= 3.5.
The default threshold sits in that gap.

This runs as the *third* stage, only on candidates that already passed the
cosine gate and the guard, so it is off the hot path for misses and cheap
rejects. It fails **open**: if the model errors or is unavailable, a candidate
that already cleared two gates is served rather than forcing every L2 hit to a
slow upstream miss. That is the opposite of the guard's ``no-text`` fail-closed,
because there the candidate was unverifiable; here only the third check is
broken.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger("cradle.rerank")


class Reranker(Protocol):
    """Minimal cross-encoder interface (satisfied by FastEmbedReranker)."""

    def score(self, query: str, candidate: str) -> float: ...


class FastEmbedReranker:
    """CPU cross-encoder over FastEmbed's TextCrossEncoder.

    Loaded once at startup (the model init + download is slow); ``score`` is a
    per-pair synchronous call meant to run inside the embed thread pool.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        *,
        cuda: bool = False,
        device_ids: list[int] | None = None,
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        kwargs: dict[str, object] = {"model_name": model_name}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if cuda:
            # FastEmbed selects CUDAExecutionProvider; requires onnxruntime-gpu
            # plus CUDA 13 / cuDNN 9 libs in the image. Fail with an actionable
            # message rather than the raw fastembed ValueError, which just says
            # "Provider CUDAExecutionProvider is not available" (issue #31): the
            # CPU image has only onnxruntime, so l2.rerank_device: cuda needs the
            # GPU image (docker/Dockerfile.gpu / docker-compose-gpu.yml).
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "l2.rerank_device is 'cuda' but onnxruntime has no "
                    f"CUDAExecutionProvider (available: {providers}; "
                    "AzureExecutionProvider ships in every CPU wheel and is inert — "
                    "the missing one is CUDA). This is the CPU image, which installs "
                    "onnxruntime, not onnxruntime-gpu. Build and run the GPU image "
                    "(docker/Dockerfile.gpu with docker-compose-gpu.yml, or "
                    "uv sync --extra rerank-gpu on a CUDA host), or set "
                    "l2.rerank_device: cpu."
                )
            kwargs["cuda"] = True
            if device_ids is not None:
                kwargs["device_ids"] = device_ids
        self._ce = TextCrossEncoder(**kwargs)  # type: ignore[arg-type]
        self.model_name = model_name

    def score(self, query: str, candidate: str) -> float:
        # rerank(query, [docs]) yields one score per doc; we pass exactly one.
        return float(next(iter(self._ce.rerank(query, [candidate]))))


def passes(
    reranker: Reranker | None,
    query_text: str,
    candidate_text: str,
    threshold: float,
) -> tuple[bool, float | None]:
    """Decide whether an L2 candidate survives the rerank stage.

    Returns ``(ok, score)``. ``ok`` is True when the candidate may be served.

    Fail-open: a missing reranker or a model error returns ``(True, None)`` so a
    candidate that already passed the cosine gate and the guard is still served.
    The caller distinguishes "served after passing rerank" (score is not None)
    from "served because rerank was unavailable" (score is None) for metrics and
    the response header.
    """
    if reranker is None:
        return True, None
    try:
        score = reranker.score(query_text, candidate_text)
    except Exception:  # noqa: BLE001 - fail open on any model/runtime error
        log.exception("rerank scoring failed; serving candidate (fail-open)")
        return True, None
    return score >= threshold, score
