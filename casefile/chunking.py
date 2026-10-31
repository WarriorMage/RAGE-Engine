"""Split a document body into chunks small enough for precise retrieval and citations.

Each chunk records the exact character range it covers in the body, so the UI can
later show and highlight the precise passage that was cited.
"""

import re
from dataclasses import dataclass

TARGET_WORDS = 140   # aim for chunks about this long
MAX_WORDS = 200      # never exceed this (unless a single sentence is longer)
MIN_WORDS = 30       # a final piece smaller than this is merged into the previous chunk


@dataclass
class Chunk:
    index: int
    text: str
    start: int   # character offset in the body (inclusive)
    end: int     # character offset in the body (exclusive)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _spans(text: str, pattern: str) -> list[tuple[int, int]]:
    """Split text on a regex separator, returning (start, end) spans of non-empty pieces."""
    spans, pos = [], 0
    for m in re.finditer(pattern, text):
        if text[pos:m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def _words(text: str, span: tuple[int, int]) -> int:
    return len(text[span[0]:span[1]].split())


def _split_big(text: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    """Break an oversized block into lines, then sentences, so no piece exceeds MAX_WORDS."""
    start, end = span
    block = text[start:end]
    for pattern in (r"\n", r"(?<=[.!?])\s+"):
        parts = [(start + s, start + e) for s, e in _spans(block, pattern)]
        if len(parts) > 1:
            out = []
            for p in parts:
                out.extend(_split_big(text, p) if _words(text, p) > MAX_WORDS else [p])
            return out
    return [span]  # cannot split further


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_text(body: str) -> list[Chunk]:
    """Pack paragraphs (blank-line separated blocks) into chunks of about TARGET_WORDS."""
    blocks: list[tuple[int, int]] = []
    for span in _spans(body, r"\n\s*\n"):
        blocks.extend(_split_big(body, span) if _words(body, span) > MAX_WORDS else [span])

    chunks: list[Chunk] = []
    cur_start = cur_end = None
    cur_words = 0
    for start, end in blocks:
        w = _words(body, (start, end))
        if cur_start is not None and (cur_words + w > MAX_WORDS or cur_words >= TARGET_WORDS):
            s, e = _trim(body, cur_start, cur_end)
            chunks.append(Chunk(len(chunks), body[s:e], s, e))
            cur_start = None
        if cur_start is None:
            cur_start, cur_words = start, 0
        cur_end = end
        cur_words += w
    if cur_start is not None:
        s, e = _trim(body, cur_start, cur_end)
        if chunks and cur_words < MIN_WORDS:
            # Too small to stand alone (e.g. a sign-off line): extend the previous chunk.
            prev = chunks[-1]
            chunks[-1] = Chunk(prev.index, body[prev.start:e], prev.start, e)
        else:
            chunks.append(Chunk(len(chunks), body[s:e], s, e))
    return chunks
