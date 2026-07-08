# RAG Vector Databases

A vector database stores embeddings and supports fast nearest-neighbor search
over them. Given a query vector it returns the stored vectors closest to it,
which is the retrieval step of a RAG pipeline.

Because exact nearest-neighbor search is slow at scale, vector databases use
approximate nearest neighbor algorithms such as HNSW, which build a navigable
graph that trades a small amount of recall for a large speedup. Each vector is
stored alongside metadata like the chunk id and source document.

Examples include Chroma, FAISS, Pinecone, and pgvector. For a small corpus you
do not strictly need one: a brute-force scan over a few thousand vectors in
plain Python is fast enough, which is why simple demos often skip the database.
