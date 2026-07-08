# TF-IDF Vectorizer

TF-IDF stands for term frequency-inverse document frequency. It turns a piece of
text into a vector of weighted term counts. The term-frequency part counts how
often each word appears in the document, and the inverse-document-frequency part
scales each word down by how common it is across the whole collection.

The effect is that words that are frequent in one document but rare overall, which
are the most distinctive, receive the highest weights. Stop words like "the" and
"and" appear everywhere and so get near-zero weight. The resulting vectors can be
compared with cosine similarity to rank documents against a query.

TF-IDF is pure counting with no learned parameters, so it runs with only the
standard library and is fully reproducible. It is the default offline retriever in
this project, standing in for a neural embedding model when no dependencies are
installed.
