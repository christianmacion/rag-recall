# Retrieval-Augmented Generation Overview

Retrieval-augmented generation, or RAG, is a pattern that combines a search step
with a language model. Instead of relying only on what the model memorized during
training, the system first retrieves relevant passages from a document store and
then asks the model to answer using those passages as context.

A RAG pipeline has four stages: chunk the documents into passages, embed each
chunk into a vector, retrieve the top-k chunks most similar to the question, and
generate an answer grounded in those chunks. The answer cites the chunk
identifiers it used.

RAG keeps answers current without retraining, lets a model work over private data,
and makes responses auditable through citations. Its weakness is that answer
quality is capped by retrieval quality: if the right chunk is never retrieved, the
model cannot use it.
