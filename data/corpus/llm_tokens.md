# LLM Tokens

A token is the basic unit of text that a large language model processes. Tokens
are not exactly words: common words may be a single token, while rare words,
code, or non-English text are split into several sub-word tokens.

Most modern tokenizers use byte-pair encoding (BPE), which merges frequently
occurring character pairs into single tokens. As a rough rule of thumb for
English prose, one token is about four characters or three-quarters of a word.

Token counts matter because they drive both cost and the context window limit.
API pricing is quoted per million input and output tokens, and a model can only
attend to a fixed maximum number of tokens at once. Counting tokens with the
model's own tokenizer is more accurate than estimating from character length.
