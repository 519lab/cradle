from __future__ import annotations

import pytest

from cradle.config import Settings
from cradle.flags import feature


def test_feature_known() -> None:
    s = Settings()
    assert feature(s, "cache") is True
    assert feature(s, "structure") is False


def test_feature_unknown() -> None:
    with pytest.raises(KeyError):
        feature(Settings(), "nope")
