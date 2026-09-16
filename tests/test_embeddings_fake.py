from __future__ import annotations

from cradle.embeddings.fake import FakeEmbedder


def test_fake_dim_and_stable() -> None:
    e = FakeEmbedder()
    a = e.embed("hello")
    b = e.embed("hello")
    c = e.embed("world")
    assert len(a) == 384
    assert a == b
    assert a != c
    assert e.ready()
