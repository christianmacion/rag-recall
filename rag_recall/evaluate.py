"""CLI scorecard: ``python -m rag_recall.evaluate``.

Loads the bundled corpus, runs the offline (stdlib TF-IDF) retriever over the
labeled eval set in ``eval/qa.jsonl``, prints a scorecard (recall@k, MRR@k,
hit-rate, plus per-answer faithfulness over the same questions), and writes
``scorecard.json``. Requires no API key and no third-party packages.

Flags:
  --k N            top-k for retrieval metrics (default 5)
  --chunk-size N   words per chunk (default from ingest)
  --overlap N      chunk overlap in words (default from ingest)
  --retriever M    tfidf | bm25 | embedding  (default tfidf; embedding needs deps)
  --out PATH       scorecard output path (default ./scorecard.json)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from . import answer as answer_mod
from . import ingest, metrics
from .retrieve import build_retriever

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QA_PATH = os.path.normpath(os.path.join(_PKG_DIR, "..", "eval", "qa.jsonl"))
DEFAULT_OUT_PATH = os.path.join(os.getcwd(), "scorecard.json")


def load_qa(path: str = DEFAULT_QA_PATH) -> List[Dict]:
    """Read the JSONL eval set: one {"q": ..., "gold_chunk_id": ...} per line."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Eval set not found: {path}")
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def run_eval(
    k: int = 3,
    chunk_size: int = ingest.DEFAULT_CHUNK_SIZE,
    overlap: int = ingest.DEFAULT_OVERLAP,
    retriever_mode: str = "tfidf",
    qa_path: str = DEFAULT_QA_PATH,
    prefer_live: bool = False,
) -> Dict:
    """Run retrieval + answer over the eval set and return a scorecard dict."""
    chunks = ingest.load_corpus(chunk_size=chunk_size, overlap=overlap)
    chunk_ids = {c.chunk_id for c in chunks}
    retriever = build_retriever(chunks, retriever_mode)
    qa = load_qa(qa_path)

    per_query = []
    per_question_rows: List[Dict] = []
    misses: List[Dict] = []
    faithfulness_sum = 0.0
    halluc_flags = 0
    unknown_gold: List[str] = []

    for row in qa:
        q = row["q"]
        gold = row["gold_chunk_id"]
        if gold not in chunk_ids:
            unknown_gold.append(gold)
        # Retrieve a generous depth so rank-of-gold is meaningful even past k.
        results = retriever.retrieve(q, max(k, 10))
        per_query.append((results, gold))

        rank = metrics.rank_of_gold(results, gold)
        hit = bool(rank and rank <= k)

        ans = answer_mod.answer_question(q, results, chunks, prefer_live=prefer_live)
        faithfulness_sum += ans.faithfulness
        halluc_flags += int(ans.hallucination_flag)

        row_out = {
            "q": q,
            "gold_chunk_id": gold,
            "gold_rank": rank,            # 0 = not retrieved at all
            "hit_at_k": hit,
            "top_result": results[0][0] if results else None,
            "faithfulness": round(ans.faithfulness, 3),
            "hallucination_flag": ans.hallucination_flag,
        }
        per_question_rows.append(row_out)
        if not hit:
            misses.append(row_out)

    card = metrics.aggregate_scorecard(per_query, k)
    n = card["n"]
    card.update(
        {
            "retriever": retriever.name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "n_chunks": len(chunks),
            "answer_mode": "live" if (prefer_live and answer_mod.live_available()) else "offline",
            "mean_faithfulness": (faithfulness_sum / n) if n else 0.0,
            "hallucination_flags": halluc_flags,
            "misses": misses,
            "per_question": per_question_rows,
            "unknown_gold_ids": unknown_gold,
        }
    )
    return card


def _print_scorecard(card: Dict) -> None:
    line = "=" * 60
    print(line)
    print("  rag-recall — RETRIEVAL SCORECARD (offline, stdlib)")
    print(line)
    print(f"  retriever        : {card['retriever']}")
    print(f"  answer mode      : {card['answer_mode']}")
    print(f"  corpus chunks    : {card['n_chunks']}  (chunk_size={card['chunk_size']}, overlap={card['overlap']})")
    print(f"  labeled questions: {card['n']}")
    print(f"  k                : {card['k']}")
    print(line)
    print(f"  recall@{card['k']}        : {card['recall_at_k']:.3f}")
    print(f"  MRR@{card['k']}           : {card['mrr_at_k']:.3f}")
    print(f"  hit-rate         : {card['hit_rate']:.3f}")
    print(f"  mean faithfulness: {card['mean_faithfulness']:.3f}")
    print(f"  hallucination fl.: {card['hallucination_flags']} / {card['n']}")
    print(line)
    if card["unknown_gold_ids"]:
        print("  WARNING: gold ids not present in corpus chunks:")
        for g in card["unknown_gold_ids"]:
            print(f"    - {g}")
        print(line)
    misses = card["misses"]
    if misses:
        print(f"  RETRIEVAL-FAILURE GALLERY ({len(misses)} miss(es) at k={card['k']}):")
        for m in misses:
            where = f"rank {m['gold_rank']}" if m["gold_rank"] else "not retrieved"
            print(f"    Q: {m['q']}")
            print(f"       gold={m['gold_chunk_id']}  ({where})  top={m['top_result']}")
    else:
        print("  No retrieval misses at this k.")
    print(line)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rag-recall retrieval scorecard")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=ingest.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=ingest.DEFAULT_OVERLAP)
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "bm25", "embedding"])
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--live", action="store_true", help="prefer live answers if ANTHROPIC_API_KEY is set")
    args = parser.parse_args(argv)

    card = run_eval(
        k=args.k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        retriever_mode=args.retriever,
        prefer_live=args.live,
    )
    _print_scorecard(card)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
