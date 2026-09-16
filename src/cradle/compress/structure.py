from __future__ import annotations


def to_dense(text: str, min_chars: int) -> str:
    """Identity in v1. Enabled only when features.structure is true (default off)."""
    _ = min_chars
    return text
