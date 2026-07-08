"""Retrieval metrics — standard library only.

All metrics are computed over a labeled set of (question, gold_chunk_id) pairs
against a retriever's ranked results. Every function here is pure and
dependency-free, so the scorecard is reproducible anywhere.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .retrieve import Result


def _ranked_ids(results: Sequence[Result]) -> List[str]:
    return [cid for cid, _ in results]


def recall_at_k(results: Sequence[Result], gold_chunk_id: str, k: int) -> float:
    """1.0 if the gold chunk appears in the top-k results, else 0.0 (per query)."""
    return 1.0 if gold_chunk_id in _ranked_ids(results)[:k] else 0.0


def reciprocal_rank_at_k(results: Sequence[Result], gold_chunk_id: str, k: int) -> float:
    """1/rank of the gold chunk within the top-k, else 0.0 (per query)."""
    ranked = _ranked_ids(results)[:k]
    for i, cid in enumerate(ranked, start=1):
        if cid == gold_chunk_id:
            return 1.0 / i
    return 0.0


def rank_of_gold(results: Sequence[Result], gold_chunk_id: str) -> int:
    """1-based rank of the gold chunk in the full result list, or 0 if absent."""
    for i, cid in enumerate(_ranked_ids(results), start=1):
        if cid == gold_chunk_id:
            return i
    return 0


def aggregate_scorecard(
    per_query: Sequence[Tuple[Sequence[Result], str]],
    k: int,
) -> dict:
    """Aggregate recall@k, MRR@k, and hit-rate over a labeled set.

    ``per_query`` is a sequence of (results, gold_chunk_id). Hit-rate is the
    same statistic as recall@k for single-gold labels; it is reported separately
    because the two diverge once a question has multiple acceptable golds.
    """
    n = len(per_query)
    if n == 0:
        return {"n": 0, "k": k, "recall_at_k": 0.0, "mrr_at_k": 0.0, "hit_rate": 0.0}

    recall_sum = 0.0
    rr_sum = 0.0
    hits = 0
    for results, gold in per_query:
        r = recall_at_k(results, gold, k)
        recall_sum += r
        rr_sum += reciprocal_rank_at_k(results, gold, k)
        hits += int(r > 0.0)

    return {
        "n": n,
        "k": k,
        "recall_at_k": recall_sum / n,
        "mrr_at_k": rr_sum / n,
        "hit_rate": hits / n,
    }
