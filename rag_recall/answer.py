"""Answer generation + faithfulness checking.

OFFLINE mode (default, no key, stdlib only): the answer is the text of the
top-retrieved chunk, returned verbatim with the cited chunk IDs. Because the
answer IS the retrieved text, it is faithful by construction — the deterministic
faithfulness check below confirms this and is the same check applied to live
answers.

LIVE mode (gated on ANTHROPIC_API_KEY): the Anthropic SDK is lazy-imported and
the model ``claude-haiku-4-5-20251001`` is asked to answer using only the
retrieved chunks. The same token-overlap faithfulness check then scores the
generated answer against the retrieved context and flags possible hallucination.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .ingest import Chunk
from .retrieve import Result, tokenize

# Model id is pinned per the project spec. Anthropic Haiku 4.5.
LIVE_MODEL = "claude-haiku-4-5-20251001"

# A claim is "grounded" if at least this fraction of its content tokens appear
# in the union of retrieved-chunk tokens.
GROUNDING_TOKEN_THRESHOLD = 0.6
# An answer is flagged as a possible hallucination below this faithfulness %.
FAITHFULNESS_FLAG_THRESHOLD = 0.7

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class AnswerResult:
    answer: str
    cited_chunk_ids: List[str]
    mode: str                       # "offline" or "live"
    faithfulness: float             # 0.0 - 1.0
    hallucination_flag: bool
    claims: List[Dict] = field(default_factory=list)  # per-claim grounding detail
    retrieved: List[Result] = field(default_factory=list)


def _split_claims(text: str) -> List[str]:
    """Split an answer into sentence-level claims, dropping trivial fragments."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if len(tokenize(s)) >= 3]


def faithfulness_check(answer_text: str, context_chunks: Sequence[Chunk]) -> Dict:
    """Deterministic token-overlap grounding check (stdlib only).

    For each sentence-level claim in the answer, measure the share of its content
    tokens that appear anywhere in the retrieved context. The faithfulness score
    is the fraction of claims that clear ``GROUNDING_TOKEN_THRESHOLD``.
    """
    context_tokens = set()
    for c in context_chunks:
        context_tokens.update(tokenize(c.text))

    claims = _split_claims(answer_text)
    if not claims:
        return {"faithfulness": 1.0, "claims": [], "hallucination_flag": False}

    detail: List[Dict] = []
    grounded_count = 0
    for claim in claims:
        ctoks = tokenize(claim)
        if not ctoks:
            continue
        overlap = sum(1 for t in ctoks if t in context_tokens)
        ratio = overlap / len(ctoks)
        is_grounded = ratio >= GROUNDING_TOKEN_THRESHOLD
        grounded_count += int(is_grounded)
        detail.append({"claim": claim, "grounding": round(ratio, 3), "grounded": is_grounded})

    faithfulness = grounded_count / len(detail) if detail else 1.0
    return {
        "faithfulness": faithfulness,
        "claims": detail,
        "hallucination_flag": faithfulness < FAITHFULNESS_FLAG_THRESHOLD,
    }


def _chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def answer_offline(
    query: str,
    results: Sequence[Result],
    chunks: Sequence[Chunk],
    top_n_context: int = 3,
) -> AnswerResult:
    """Deterministic extractive answer: return the top chunk text + cited ids."""
    by_id = _chunks_by_id(chunks)
    cited = [cid for cid, _ in results[:top_n_context]]
    context = [by_id[cid] for cid in cited if cid in by_id]

    if not context:
        return AnswerResult(
            answer="No relevant passage was retrieved for this question.",
            cited_chunk_ids=[],
            mode="offline",
            faithfulness=1.0,
            hallucination_flag=False,
            claims=[],
            retrieved=list(results),
        )

    top = context[0]
    answer_text = top.text
    fc = faithfulness_check(answer_text, context)
    return AnswerResult(
        answer=answer_text,
        cited_chunk_ids=cited,
        mode="offline",
        faithfulness=fc["faithfulness"],
        hallucination_flag=fc["hallucination_flag"],
        claims=fc["claims"],
        retrieved=list(results),
    )


def answer_live(
    query: str,
    results: Sequence[Result],
    chunks: Sequence[Chunk],
    top_n_context: int = 3,
) -> AnswerResult:
    """Generate an answer with Anthropic Haiku 4.5, grounded in retrieved chunks.

    Lazy-imports the Anthropic SDK. Requires ANTHROPIC_API_KEY. The same
    faithfulness check scores the model output against the retrieved context.
    """
    try:
        import anthropic  # noqa: WPS433  (lazy import — never on the offline path)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Live mode requires the 'anthropic' package. Install it (see "
            "requirements.txt) or use offline mode."
        ) from exc

    by_id = _chunks_by_id(chunks)
    cited = [cid for cid, _ in results[:top_n_context]]
    context = [by_id[cid] for cid in cited if cid in by_id]

    context_block = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in context)
    system = (
        "You answer questions using ONLY the provided context passages. "
        "Cite the chunk ids you used in square brackets. If the context does not "
        "contain the answer, say so plainly. Do not use outside knowledge."
    )
    user = f"Context passages:\n\n{context_block}\n\nQuestion: {query}"

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=LIVE_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer_text = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"), ""
    ).strip()

    fc = faithfulness_check(answer_text, context)
    return AnswerResult(
        answer=answer_text,
        cited_chunk_ids=cited,
        mode="live",
        faithfulness=fc["faithfulness"],
        hallucination_flag=fc["hallucination_flag"],
        claims=fc["claims"],
        retrieved=list(results),
    )


def live_available() -> bool:
    """True only if a key is set AND the anthropic SDK is importable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401,WPS433
    except ImportError:
        return False
    return True


def answer_question(
    query: str,
    results: Sequence[Result],
    chunks: Sequence[Chunk],
    prefer_live: bool = False,
    top_n_context: int = 3,
) -> AnswerResult:
    """Dispatch to live mode if requested AND available, else offline."""
    if prefer_live and live_available():
        return answer_live(query, results, chunks, top_n_context)
    return answer_offline(query, results, chunks, top_n_context)
