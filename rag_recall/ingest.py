"""Corpus loading and chunking — standard library only.

Produces a flat list of chunks with deterministic IDs of the form
``<docname>#<chunk_index>`` (e.g. ``rag_chunking#0``). The doc name is the
markdown filename without its ``.md`` extension. IDs are stable as long as the
file contents and the chunk size do not change, which is what makes the gold
labels in ``eval/qa.jsonl`` line up with what retrieval actually produces.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

# Default location of the bundled corpus, resolved relative to this file so the
# package works regardless of the current working directory.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.normpath(os.path.join(_PKG_DIR, "..", "data", "corpus"))

# Chunking defaults. Chunk size is measured in words. Documents in this corpus
# are short (one topic each), so the default size keeps most docs to one or two
# chunks while still exercising the chunk-index part of the ID scheme.
DEFAULT_CHUNK_SIZE = 90
DEFAULT_OVERLAP = 20


@dataclass(frozen=True)
class Chunk:
    """A single retrievable passage."""

    chunk_id: str          # e.g. "rag_chunking#0"
    doc: str               # e.g. "rag_chunking"
    index: int             # chunk index within the doc
    text: str              # the chunk's raw text


_WORD_RE = re.compile(r"\S+")


def _split_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word windows.

    A single trailing window shorter than ``chunk_size`` is kept as its own
    chunk. Overlap is clamped so it can never stall the loop.
    """
    words = _split_words(text)
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0
    n = len(words)
    while start < n:
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= n:
            break
        start += step
    return chunks


def load_corpus(
    corpus_dir: str = CORPUS_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Load every ``*.md`` file in ``corpus_dir`` and chunk it deterministically.

    Files are processed in sorted filename order so chunk IDs are stable across
    runs and machines.
    """
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    chunks: List[Chunk] = []
    filenames = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".md"))
    for filename in filenames:
        doc = filename[:-3]  # strip ".md"
        path = os.path.join(corpus_dir, filename)
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        for i, piece in enumerate(chunk_text(raw, chunk_size, overlap)):
            chunks.append(Chunk(chunk_id=f"{doc}#{i}", doc=doc, index=i, text=piece))
    return chunks
