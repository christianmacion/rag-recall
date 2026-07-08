# RAG Embeddings

An embedding is a dense vector of numbers that represents the meaning of a piece
of text. Texts with similar meaning map to nearby vectors, so semantic similarity
becomes a distance or angle between vectors. Cosine similarity is the most common
measure.

In a RAG system both the chunks and the user's question are embedded with the
same model, and retrieval finds the chunks whose vectors are closest to the
question vector. A popular open embedding model is all-MiniLM-L6-v2, which
produces 384-dimensional vectors.

Embeddings capture meaning beyond exact keyword overlap, so a question phrased
differently from the source can still match. The trade-off is that a pure
keyword method like BM25 can be stronger when the answer hinges on a rare exact
term such as an identifier or error code.
