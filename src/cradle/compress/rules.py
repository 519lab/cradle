from __future__ import annotations

import re

LEADING = re.compile(
    r"^(hi|hello|hey|please|thanks|thank you|could you|would you|can you)\b[ ,.!]*",
    re.I,
)
TRAILING = re.compile(
    r"[ ,.!]*\b(thanks|thank you|cheers|regards|please)\b[ ,.!]*$",
    re.I,
)
FILLERS = (
    re.compile(r"\bjust\b", re.I),
    re.compile(r"\breally\b", re.I),
    re.compile(r"\bvery\b", re.I),
    re.compile(r"\bactually\b", re.I),
    re.compile(r"\bbasically\b", re.I),
    re.compile(r"\bsimply\b", re.I),
    re.compile(r"\bkind of\b", re.I),
    re.compile(r"\bsort of\b", re.I),
)
BANGS = re.compile(r"!{2,}")


def _pass(text: str) -> str:
    """One application of every rule, on whitespace-collapsed text."""
    out = " ".join(text.split())
    out = LEADING.sub("", out, count=1)
    out = TRAILING.sub("", out, count=1)
    for pat in FILLERS:
        out = pat.sub("", out)
    out = BANGS.sub("!", out)
    return " ".join(out.split()).strip()


def strip_fluff(text: str) -> str:
    """Apply the fluff rules until nothing changes.

    A single ordered pass is not idempotent: LEADING/TRAILING are anchored at
    ^/$ and only skip [ ,.!], so leading whitespace or a newline around a
    closing pleasantry hid it, and removing a filler ("just please …") can
    expose a leading pleasantry the earlier rule already ran past. Iterating to
    a fixed point makes the result independent of rule order. It terminates
    because every rule only removes text (tests/test_properties.py pins both
    idempotence and never-grows).
    """
    out = text
    while True:
        nxt = _pass(out)
        if nxt == out:
            return out
        out = nxt
