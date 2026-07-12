"""Chunking: page-tagged text blocks -> retrieval-sized chunks.

Phase 1 chunks per-page with a sliding token window. Structure-aware
chunking (heading boundaries, never splitting mid-table) needs the layout
info that Docling provides — deferred until that parser is wired in.
"""

from dataclasses import dataclass

import tiktoken

from docuagent.ingest.parse import TextBlock

_ENCODING = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 650
OVERLAP_RATIO = 0.15


@dataclass
class Chunk:
    text: str
    page: int
    section: str | None
    chunk_index: int


def _split_text(text: str) -> list[str]:
    tokens = _ENCODING.encode(text)
    if len(tokens) <= TARGET_TOKENS:
        return [text]

    step = max(1, int(TARGET_TOKENS * (1 - OVERLAP_RATIO)))
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + TARGET_TOKENS, len(tokens))
        pieces.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return pieces


def chunk_blocks(blocks: list[TextBlock]) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for block in blocks:
        for piece in _split_text(block.text):
            chunks.append(Chunk(text=piece, page=block.page, section=block.section, chunk_index=index))
            index += 1
    return chunks
