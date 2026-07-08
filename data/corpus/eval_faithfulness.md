# Evaluation: Faithfulness

Faithfulness measures whether an answer's claims are actually supported by the
retrieved context, as opposed to invented by the model. An answer can be fluent
and even correct yet still be unfaithful if it states things the retrieved chunks
do not back up.

A simple faithfulness check breaks the answer into claims and tests each one
against the retrieved passages. The faithfulness percentage is the share of claims
that are grounded. When that share falls below a threshold, the answer is flagged
as a possible hallucination.

Faithfulness is distinct from answer correctness and from retrieval recall. You
can retrieve the right chunk, recall@k of one, and still generate an unfaithful
answer; conversely a faithful answer is only as good as the chunk it was grounded
in. Tracking faithfulness alongside recall gives a fuller picture of RAG quality.
