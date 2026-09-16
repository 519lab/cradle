from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cradle.config import Settings


def feature(settings: Settings, name: str) -> bool:
    flags = settings.features
    value = getattr(flags, name, None)
    if value is None:
        raise KeyError(f"unknown feature flag: {name}")
    return bool(value)
