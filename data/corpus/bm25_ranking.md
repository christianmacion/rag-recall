# BM25 Ranking

BM25 is a classic ranking function used to score how well a document matches a
search query based on the words they share. It builds on term frequency and
inverse document frequency, rewarding documents that contain the query terms
often while down-weighting terms that appear in many documents.

Two parameters tune BM25. The k1 parameter controls term-frequency saturation, so
that the tenth occurrence of a word adds less than the second. The b parameter
controls length normalization, which prevents long documents from scoring high
just because they contain more words overall.

BM25 needs no training and no embeddings, which makes it a strong, fully offline
retrieval baseline. It excels when the answer depends on exact rare terms, but it
cannot match a question to a passage that uses entirely different wording, which
is where dense embeddings help.
