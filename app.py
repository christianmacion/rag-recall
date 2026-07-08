"""rag-recall — Streamlit UI.

Two views:
  * Ask — type a question, see the cited chunks and the (offline or live) answer
    plus its faithfulness score.
  * Run eval — render the retrieval scorecard over the bundled labeled set, with
    a retrieval-failure gallery.

k and chunk-size/overlap are adjustable in the sidebar. Everything runs offline
with no API key; if ANTHROPIC_API_KEY is set and the anthropic SDK is installed,
an "Use live answers" toggle becomes available.

Author: Christian Macion — AI / Agent Engineer.
"""

from __future__ import annotations

import streamlit as st

from rag_recall import answer as answer_mod
from rag_recall import ingest
from rag_recall.evaluate import run_eval
from rag_recall.retrieve import build_retriever

st.set_page_config(page_title="rag-recall", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def _load(chunk_size: int, overlap: int, retriever_mode: str):
    """Build (and cache) the corpus + retriever for a given config."""
    chunks = ingest.load_corpus(chunk_size=chunk_size, overlap=overlap)
    retriever = build_retriever(chunks, retriever_mode)
    return chunks, retriever


def _chunk_lookup(chunks):
    return {c.chunk_id: c for c in chunks}


# ---- Sidebar controls -------------------------------------------------------
st.sidebar.title("rag-recall")
st.sidebar.caption("Document Q&A that proves its own retrieval.")

k = st.sidebar.slider("Top-k retrieved", min_value=1, max_value=10, value=3)
chunk_size = st.sidebar.slider("Chunk size (words)", 40, 160, ingest.DEFAULT_CHUNK_SIZE, step=10)
overlap = st.sidebar.slider("Chunk overlap (words)", 0, 60, ingest.DEFAULT_OVERLAP, step=5)

retriever_mode = "tfidf"
emb_choice = st.sidebar.selectbox(
    "Retriever",
    options=["tfidf (stdlib, offline)", "bm25 (stdlib, offline)", "embedding (needs deps)"],
    index=0,
)
retriever_mode = {"tfidf": "tfidf", "bm25": "bm25", "embed": "embedding"}[emb_choice.split()[0][:5]]

live_ok = answer_mod.live_available()
prefer_live = False
if live_ok:
    prefer_live = st.sidebar.toggle("Use live answers (Anthropic Haiku 4.5)", value=False)
    st.sidebar.success("Live mode available (ANTHROPIC_API_KEY detected).")
else:
    st.sidebar.info("Offline mode — no ANTHROPIC_API_KEY. Answers are extractive.")

try:
    chunks, retriever = _load(chunk_size, overlap, retriever_mode)
except ImportError as exc:
    st.sidebar.error(str(exc))
    st.stop()

by_id = _chunk_lookup(chunks)

tab_ask, tab_eval = st.tabs(["Ask", "Run eval"])

# ---- Ask tab ----------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a question over the bundled corpus")
    default_q = "What is retrieval-augmented generation and what are its four stages?"
    query = st.text_input("Question", value=default_q)
    go = st.button("Retrieve & answer", type="primary")

    if go and query.strip():
        results = retriever.retrieve(query, max(k, 10))
        ans = answer_mod.answer_question(
            query, results, chunks, prefer_live=prefer_live, top_n_context=min(3, k)
        )

        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Answer** ({ans.mode} mode)")
            st.write(ans.answer if ans.answer else "_(no answer)_")
            st.markdown("**Cited chunks:** " + ", ".join(f"`{c}`" for c in ans.cited_chunk_ids))

            color = "🟢" if not ans.hallucination_flag else "🔴"
            st.metric("Faithfulness", f"{ans.faithfulness * 100:.0f}%")
            st.caption(
                f"{color} "
                + ("grounded in retrieved context" if not ans.hallucination_flag
                   else "possible hallucination — claims not fully supported")
            )
            if ans.claims:
                with st.expander("Per-claim grounding"):
                    for cl in ans.claims:
                        mark = "✅" if cl["grounded"] else "⚠️"
                        st.write(f"{mark} ({cl['grounding']:.2f}) {cl['claim']}")

        with right:
            st.markdown(f"**Top-{k} retrieved chunks**")
            for rank, (cid, score) in enumerate(results[:k], start=1):
                chunk = by_id.get(cid)
                if not chunk:
                    continue
                with st.expander(f"#{rank} · `{cid}` · score {score:.3f}"):
                    st.write(chunk.text)

# ---- Eval tab ---------------------------------------------------------------
with tab_eval:
    st.subheader("Retrieval scorecard")
    st.caption(
        "Computed over the in-repo labeled set (eval/qa.jsonl) with the current "
        "sidebar settings. Reproduce on the CLI: `python -m rag_recall.evaluate`."
    )
    if st.button("Run eval", type="primary"):
        card = run_eval(
            k=k, chunk_size=chunk_size, overlap=overlap,
            retriever_mode=retriever_mode, prefer_live=prefer_live,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"recall@{card['k']}", f"{card['recall_at_k']:.3f}")
        c2.metric(f"MRR@{card['k']}", f"{card['mrr_at_k']:.3f}")
        c3.metric("hit-rate", f"{card['hit_rate']:.3f}")
        c4.metric("mean faithfulness", f"{card['mean_faithfulness']:.3f}")

        st.caption(
            f"{card['n']} labeled questions · {card['n_chunks']} chunks · "
            f"{card['retriever']} · {card['answer_mode']} answers · "
            f"{card['hallucination_flags']} hallucination flag(s)"
        )

        misses = card["misses"]
        st.markdown(f"### Retrieval-failure gallery ({len(misses)} miss(es) at k={card['k']})")
        if not misses:
            st.success("No retrieval misses at this k.")
        for m in misses:
            where = f"buried at rank {m['gold_rank']}" if m["gold_rank"] else "not retrieved at all"
            st.warning(
                f"**Q:** {m['q']}\n\n"
                f"gold `{m['gold_chunk_id']}` was {where}; top result was `{m['top_result']}`."
            )

        with st.expander("Full per-question table"):
            st.dataframe(card["per_question"], use_container_width=True)

st.divider()
st.caption("Christian Macion — AI / Agent Engineer · rag-recall")
