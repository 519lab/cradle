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


def strip_fluff(text: str) -> str:
    out = text
    while True:
        nxt = LEADING.sub("", out, count=1)
        if nxt == out:
            break
        out = nxt
    while True:
        nxt = TRAILING.sub("", out, count=1)
        if nxt == out:
            break
        out = nxt
    for pat in FILLERS:
        out = pat.sub("", out)
    out = BANGS.sub("!", out)
    return " ".join(out.split()).strip()
