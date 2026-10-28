from __future__ import annotations

import json
import re
from dataclasses import dataclass

PLACEHOLDER_FMT = "\x00CRADLE_KEEP_{n:03d}\x00"

FORMAT_LINE_PATTERNS = (
    re.compile(r"format your answer as", re.I),
    re.compile(r"respond (only )?in json", re.I),
    re.compile(r"output (only )?valid json", re.I),
    re.compile(r"you must (reply|respond|answer) (in|with)", re.I),
    re.compile(r"return (only )?a (json|yaml|csv|table)", re.I),
    re.compile(r"do not (include|add) (markdown|commentary)", re.I),
)


@dataclass
class Span:
    start: int
    end: int
    text: str
    reason: str


def _overlaps(covered: list[tuple[int, int]], start: int, end: int) -> bool:
    for a, b in covered:
        if start < b and end > a:
            return True
    return False


def _add(covered: list[tuple[int, int]], start: int, end: int) -> None:
    covered.append((start, end))


def _fence_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    i = 0
    n = len(text)
    while i < n:
        line_start = i
        nl = text.find("\n", i)
        line_end = n if nl < 0 else nl
        line = text[line_start:line_end]
        m = re.match(r"^(\s*)(```|~~~)([^\n]*)\s*$", line)
        if not m:
            i = line_end + 1 if nl >= 0 else n
            continue
        fence = m.group(2)
        opener_end = line_end + 1 if nl >= 0 else n
        close = re.search(rf"^{re.escape(m.group(1))}{re.escape(fence)}[^\n]*$", text[opener_end:], re.M)
        if close:
            end = opener_end + close.end()
            if opener_end + close.end() < n and text[opener_end + close.end() : opener_end + close.end() + 1] == "\n":
                end += 1
        else:
            end = n
        spans.append(Span(line_start, end, text[line_start:end], "code"))
        i = end
    return spans


def _inline_tick_spans(text: str, covered: list[tuple[int, int]]) -> list[Span]:
    spans: list[Span] = []
    for m in re.finditer(r"`[^`]+`", text):
        if _overlaps(covered, m.start(), m.end()):
            continue
        spans.append(Span(m.start(), m.end(), m.group(0), "code"))
    return spans


def _json_spans(text: str, covered: list[tuple[int, int]]) -> list[Span]:
    spans: list[Span] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        if _overlaps(covered, i, i + 1):
            i += 1
            continue
        try:
            _obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        raw = text[i : i + end]
        if len(raw) >= 8 and not _overlaps(covered, i, i + end):
            spans.append(Span(i, i + end, raw, "json"))
            i += end
            continue
        i += 1
    return spans


def _format_spans(text: str, covered: list[tuple[int, int]]) -> list[Span]:
    spans: list[Span] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        start = offset
        end = offset + len(line)
        offset = end
        body = line.strip("\n")
        if not body.strip():
            continue
        if _overlaps(covered, start, end):
            continue
        if any(p.search(body) for p in FORMAT_LINE_PATTERNS):
            spans.append(Span(start, end, line, "format"))
    return spans


def extract_protected(text: str) -> tuple[list[str], str, list[Span]]:
    covered: list[tuple[int, int]] = []
    spans: list[Span] = []
    for s in _fence_spans(text):
        if not _overlaps(covered, s.start, s.end):
            spans.append(s)
            _add(covered, s.start, s.end)
    for s in _inline_tick_spans(text, covered):
        spans.append(s)
        _add(covered, s.start, s.end)
    for s in _json_spans(text, covered):
        spans.append(s)
        _add(covered, s.start, s.end)
    for s in _format_spans(text, covered):
        spans.append(s)
        _add(covered, s.start, s.end)
    spans.sort(key=lambda s: s.start)
    placeholders: list[str] = []
    masked = text
    for i, s in enumerate(reversed(spans)):
        n = len(spans) - 1 - i
        token = PLACEHOLDER_FMT.format(n=n)
        placeholders.insert(0, s.text)
        masked = masked[: s.start] + token + masked[s.end :]
    return placeholders, masked, spans


def restore_protected(text: str, placeholders: list[str]) -> str:
    out = text
    for n in range(len(placeholders) - 1, -1, -1):
        out = out.replace(PLACEHOLDER_FMT.format(n=n), placeholders[n])
    return out


def format_instruction_lines(spans: list[Span]) -> list[str]:
    return [s.text.strip() for s in spans if s.reason == "format"]
