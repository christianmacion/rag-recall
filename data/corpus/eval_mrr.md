# Evaluation: Mean Reciprocal Rank

Mean reciprocal rank, or MRR, measures not just whether the gold chunk was
retrieved but how high it was ranked. For one question the reciprocal rank is one
divided by the position of the first relevant result: the gold chunk at rank one
scores 1.0, at rank two scores 0.5, at rank three scores about 0.33, and zero if
it is not retrieved at all.

MRR@k is the mean of these reciprocal ranks across all labeled questions,
considering only the top-k positions. It rewards systems that put the right
passage near the top, which matters because the generator weighs earlier chunks
more and the context budget is limited.

MRR and recall@k are complementary: recall asks whether the chunk was found, MRR
asks how well it was ranked. A system can have high recall but mediocre MRR if it
retrieves the right chunk but buries it low in the list.
