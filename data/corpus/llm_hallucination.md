# LLM Hallucination

A hallucination is when a language model produces text that is fluent and
confident but factually wrong or unsupported by any source. The model is
optimizing for plausible-sounding continuations, not for truth, so it can invent
citations, dates, or quotations that never existed.

Hallucinations are more likely on topics outside the training data, on very
recent events, and when the prompt pressures the model to answer rather than
admit uncertainty. They are a central risk for any application that presents
model output as fact.

Retrieval-augmented generation reduces hallucination by grounding answers in
retrieved source passages and citing them, so claims can be checked. A
faithfulness check compares each claim in the answer against the retrieved
context and flags any claim that is not supported.
