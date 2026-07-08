# RAG Chunking

Chunking is the step that splits source documents into smaller passages before
embedding. Chunk size is a key tuning knob: chunks that are too large dilute the
embedding with unrelated content and waste context, while chunks that are too
small lose the surrounding meaning needed to answer a question.

A common strategy is fixed-size chunking with overlap, where each chunk shares a
few sentences with its neighbor so an idea split across a boundary still appears
intact in at least one chunk. Semantic chunking instead splits on natural
boundaries like paragraphs or headings.

Each chunk is given a stable identifier so the answer can cite exactly which
passages it used. In this project a chunk id is formed as the document name
followed by a hash and the chunk index, for example `rag_chunking#0`.
