# Evaluation: Recall at K

Recall@k measures whether retrieval found the right passage. For each labeled
question there is a gold chunk that contains the answer. Recall@k is the fraction
of questions for which the gold chunk appears anywhere in the top-k retrieved
results.

Recall@k rises as k increases, because retrieving more chunks gives more chances
to include the gold one. A high recall@k is necessary but not sufficient for a
good RAG system: the generator still has to use the retrieved chunk correctly.

Recall@k is the single most important retrieval metric to track, because a chunk
that is never retrieved can never be cited. It is computed over a held-out labeled
set of question and gold-chunk pairs, not on the training corpus.
