"""rag-recall — Document Q&A that proves its own retrieval.

A small, dependency-light RAG service. The offline path (chunk -> TF-IDF embed ->
retrieve -> extractive answer -> faithfulness check) runs on the Python standard
library alone, so every metric is reproducible with zero installs. An optional
live path uses sentence-transformers + chromadb for retrieval and the Anthropic
SDK for answer generation when those are installed and a key is present.

Author: Christian Macion — AI / Agent Engineer.
"""

__all__ = ["ingest", "retrieve", "answer", "metrics", "evaluate"]
__version__ = "0.1.0"
