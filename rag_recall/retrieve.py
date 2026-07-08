"""Retrieval.

The DEFAULT, fully-offline retriever is a pure-Python TF-IDF vectorizer with
cosine similarity (``TfidfRetriever``). It needs nothing beyond the standard
library — no numpy, no sentence-transformers — which is what makes recall@k
reproducible anywhere. A BM25 retriever (``Bm25Retriever``) is also provided as
a stdlib alternative.

An OPTIONAL high-quality retriever (``EmbeddingRetriever``) uses
sentence-transformers (all-MiniLM-L6-v2) + chromadb. Those imports are lazy and
happen only inside that class, so importing this module never pulls them in. The
offline code path never touches them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .ingest import Chunk

# A retrieval result: (chunk_id, score), best first.
Result = Tuple[str, float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, generic English stop-word set. Kept inline so the offline path has no
# data-file dependency.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how in into is it its of on or
    that the their them then there these they this to was were what when where
    which who will with you your
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer that drops stop words and 1-char tokens."""
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller dict for the dot product.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(term, 0.0) for term, w in a.items())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class TfidfRetriever:
    """Pure-Python TF-IDF retriever with cosine similarity (offline default)."""

    name = "tfidf (stdlib, offline)"

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks: List[Chunk] = list(chunks)
        self._doc_tokens: List[List[str]] = [tokenize(c.text) for c in self.chunks]
        self._idf: Dict[str, float] = self._compute_idf(self._doc_tokens)
        self._vectors: List[Dict[str, float]] = [
            self._tfidf_vector(toks) for toks in self._doc_tokens
        ]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def _compute_idf(self, docs: List[List[str]]) -> Dict[str, float]:
        n = len(docs)
        df: Counter = Counter()
        for toks in docs:
            for term in set(toks):
                df[term] += 1
        # Smoothed idf; always positive so every observed term carries weight.
        return {term: math.log((1 + n) / (1 + d)) + 1.0 for term, d in df.items()}

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            vec[term] = (count / total) * idf
        return vec

    def retrieve(self, query: str, k: int) -> List[Result]:
        qvec = self._tfidf_vector(tokenize(query))
        scored: List[Result] = []
        for chunk, dvec in zip(self.chunks, self._vectors):
            score = _cosine(qvec, dvec)
            if score > 0.0:
                scored.append((chunk.chunk_id, score))
        # Stable tie-break on chunk_id keeps results deterministic.
        scored.sort(key=lambda r: (-r[1], r[0]))
        return scored[:k]


class Bm25Retriever:
    """Pure-Python BM25 retriever (offline alternative)."""

    name = "bm25 (stdlib, offline)"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks: List[Chunk] = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_tokens: List[List[str]] = [tokenize(c.text) for c in self.chunks]
        self._doc_len: List[int] = [len(t) for t in self._doc_tokens]
        n = len(self.chunks)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        df: Counter = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] += 1
        self._idf: Dict[str, float] = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }
        self._tf: List[Counter] = [Counter(t) for t in self._doc_tokens]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, k: int) -> List[Result]:
        q_terms = set(tokenize(query))
        scored: List[Result] = []
        for chunk, tf, dl in zip(self.chunks, self._tf, self._doc_len):
            score = 0.0
            for term in q_terms:
                idf = self._idf.get(term)
                if idf is None:
                    continue
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0)
                )
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0.0:
                scored.append((chunk.chunk_id, score))
        scored.sort(key=lambda r: (-r[1], r[0]))
        return scored[:k]


class EmbeddingRetriever:
    """OPTIONAL dense retriever: sentence-transformers + chromadb.

    Lazy-imported. Only constructed by callers that explicitly request the live
    retriever AND have the optional dependencies installed. The offline path
    never instantiates this class, so the heavy imports never run there.
    """

    name = "all-MiniLM-L6-v2 + chroma (live)"

    def __init__(self, chunks: Sequence[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
            import chromadb  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - exercised only when missing
            raise ImportError(
                "EmbeddingRetriever requires 'sentence-transformers' and "
                "'chromadb'. Install them (see requirements.txt) or use the "
                "offline TfidfRetriever."
            ) from exc

        self.chunks: List[Chunk] = list(chunks)
        self._model = SentenceTransformer(model_name)
        client = chromadb.EphemeralClient()
        # Fresh collection each run keeps the demo stateless.
        self._collection = client.create_collection(name="rag_recall")
        embeddings = self._model.encode(
            [c.text for c in self.chunks], show_progress_bar=False
        ).tolist()
        self._collection.add(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=embeddings,
            documents=[c.text for c in self.chunks],
        )

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, k: int) -> List[Result]:
        q_emb = self._model.encode([query], show_progress_bar=False).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=k)
        ids = res["ids"][0]
        distances = res["distances"][0]
        # Chroma returns squared L2 distance; convert to a descending similarity.
        return [(cid, 1.0 / (1.0 + dist)) for cid, dist in zip(ids, distances)]


def build_retriever(chunks: Sequence[Chunk], mode: str = "tfidf"):
    """Factory. ``mode`` is one of 'tfidf', 'bm25', or 'embedding'.

    Only 'embedding' touches the optional dependencies.
    """
    mode = mode.lower()
    if mode == "tfidf":
        return TfidfRetriever(chunks)
    if mode == "bm25":
        return Bm25Retriever(chunks)
    if mode == "embedding":
        return EmbeddingRetriever(chunks)
    raise ValueError(f"Unknown retriever mode: {mode!r}")
