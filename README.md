# rag-recall — Document Q&A that proves its own retrieval

**TL;DR** — A small RAG service that answers questions over a bundled doc set
**and ships a retrieval scorecard**, so you can see whether retrieval actually
found the right passage instead of taking the answer on faith.

> **Headline metric: recall@3 = 0.886, MRR@3 = 0.805 on 35 labeled questions**
> (offline stdlib TF-IDF retriever over a 17-doc / 34-chunk corpus; mean
> faithfulness 1.00, 0 hallucination flags). Reproduce in one command — no API
> key, no pip installs.

```
python3 -m rag_recall.evaluate
```

**How a reviewer clicks it:** open the app, hit **Run eval** to see the live
scorecard, then go to the **Ask** tab, type a question, and watch the cited
chunks and faithfulness score update.

---

## Why this exists

Most RAG demos show you an answer and stop there. The honest question is: *did
retrieval even surface the passage the answer should come from?* rag-recall
answers that with a reproducible scorecard:

- **recall@k** — did the gold chunk land in the top-k?
- **MRR@k** — how highly was it ranked?
- **hit-rate** — fraction of questions with the gold chunk retrieved.
- **faithfulness %** + **hallucination flag** — are the answer's claims actually
  grounded in the retrieved chunks?

All four are computed over an in-repo, hand-labeled set of 35 (question,
gold_chunk_id) pairs in `eval/qa.jsonl`.

---

## Offline vs live

| | Offline (default) | Live (opt-in) |
|---|---|---|
| **Retrieval** | Pure-Python TF-IDF (or BM25), cosine similarity | optional `sentence-transformers` (all-MiniLM-L6-v2) + `chromadb` |
| **Answer** | Deterministic extractive (top chunk + cited ids) | Anthropic `claude-haiku-4-5-20251001`, grounded in retrieved chunks |
| **Faithfulness** | Deterministic token-overlap grounding check | same check, applied to the model's output |
| **Needs** | Python stdlib only — **no installs, no key** | `ANTHROPIC_API_KEY` + the optional packages |

The offline path renders **every metric** with no key and no third-party
packages. The heavy imports (`anthropic`, `sentence-transformers`, `chromadb`)
are **lazy** — they live inside the live/embedding code paths and are never
imported on the offline path. The eval CLI never touches them.

Live mode activates only when `ANTHROPIC_API_KEY` is set **and** the `anthropic`
SDK is importable; otherwise the app silently stays offline.

---

## Run it locally

Offline eval (stdlib only — works out of the box):

```bash
cd rag-recall
python3 -m rag_recall.evaluate                 # prints scorecard, writes scorecard.json
python3 -m rag_recall.evaluate --k 5           # toggle k
python3 -m rag_recall.evaluate --chunk-size 60 --overlap 10
python3 -m rag_recall.evaluate --retriever bm25
```

The app (needs `streamlit`):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live answers (optional): `export ANTHROPIC_API_KEY=sk-ant-...` before launching
the app, then flip the **Use live answers** toggle in the sidebar.

---

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing
   at `app.py` on your branch.
3. (Optional) add `ANTHROPIC_API_KEY` under **Settings → Secrets** to enable live
   answers. Without it the app runs fully offline — every metric still renders.

`requirements.txt` and `.streamlit/config.toml` (theme primary `#0b3d5c`) are
already in place.

---

## The one honest metric (reproducible)

`python3 -m rag_recall.evaluate` is the single command behind the headline. It
loads the corpus, runs the offline TF-IDF retriever over `eval/qa.jsonl`, prints
the scorecard, and writes `scorecard.json`. Current output:

```
recall@3        : 0.886
MRR@3           : 0.805
hit-rate        : 0.886
mean faithfulness: 1.000
hallucination fl.: 0 / 35
```

Chunk IDs are deterministic — `<docname>#<chunk_index>` (e.g. `rag_chunking#1`) —
so the gold labels in `eval/qa.jsonl` line up exactly with what ingest/retrieve
produce. Change `--chunk-size` and the IDs (and gold alignment) shift; the
default settings are the ones the labels were written against.

---

## Honest scope

- **Tiny, self-authored corpus** (17 short notes on Python / LLM / RAG / eval
  concepts). Recall is high partly *because* the corpus and questions were
  written together — this is a methodology demo, not a benchmark on adversarial
  data. The 4 genuine misses below are the honest part.
- **Offline answers are extractive**, not abstractive — they return the
  top-retrieved chunk verbatim, so they are faithful by construction. The
  faithfulness check is most interesting in **live** mode, where it scores a
  generated answer that could drift from its sources.
- **Faithfulness is a token-overlap proxy**, not an entailment model. It catches
  ungrounded claims by lexical overlap; it won't catch a paraphrase that is
  semantically wrong but lexically close. It's a cheap, deterministic guardrail,
  not a judge.
- **TF-IDF / BM25 are keyword retrievers.** They miss when a question is phrased
  with entirely different vocabulary than the source — exactly the failure gallery
  below. The optional embedding retriever exists to close that gap.

### Retrieval-failure gallery (the 4 misses at k=3)

| Question (paraphrased away from source wording) | Gold chunk | What happened | Root cause |
|---|---|---|---|
| "…ranks it third instead of first; which metric captures that ranking quality?" | `eval_mrr#0` | gold buried at rank 8; `bm25_ranking#0` won | "ranks third" shares no terms with "reciprocal rank" — pure vocabulary mismatch |
| "How can I avoid loading an entire million-item sequence into memory all at once?" | `python_generators#1` | gold at rank 9; `python_generators#0` won | right doc, wrong chunk — "memory" is densest in chunk #0, so the generator-expression fact in #1 loses |
| "How do I reproduce the exact set of packages another developer installed?" | `python_virtual_environments#1` | gold at rank 6; `…#0` won | the `requirements.txt` fact lives in #1, but "packages/install" vocabulary is heavier in #0 |
| "What stops the tenth occurrence of a word from counting as much as the second occurrence?" | `bm25_ranking#1` | gold at rank 5; `bm25_ranking#0` won | the k1 / term-frequency-saturation answer is in #1; the question's plain wording matches the #0 overview better |

Three of the four are **same-document, wrong-chunk** failures — the classic
chunking trade-off — and one is a clean **vocabulary mismatch** that dense
embeddings would likely fix. That's the point of shipping the scorecard: the
misses are visible and explainable, not hidden behind a confident answer.

---

## Layout

```
app.py                      Streamlit UI (Ask + Run eval, k/chunk toggles)
rag_recall/
  __init__.py
  ingest.py                 corpus load + deterministic chunking (<doc>#<i>)
  retrieve.py               TF-IDF (default) + BM25 + optional embedding retriever
  answer.py                 extractive (offline) / Anthropic Haiku (live) + faithfulness
  metrics.py                recall@k, MRR@k, hit-rate (stdlib, pure)
  evaluate.py               CLI scorecard -> stdout + scorecard.json
data/corpus/*.md            17 short factual notes
eval/qa.jsonl               35 labeled (q, gold_chunk_id) pairs
requirements.txt            UI + live/embedding deps only (eval needs none)
.streamlit/config.toml      theme primaryColor #0b3d5c
```

---

*Christian Macion — AI / Agent Engineer.*
