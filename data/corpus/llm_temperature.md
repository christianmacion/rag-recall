# LLM Temperature and Sampling

Temperature is a sampling parameter that controls the randomness of a language
model's output. A low temperature near zero makes the model nearly deterministic,
favoring the highest-probability next token, while a higher temperature flattens
the distribution and produces more varied, creative text.

Top-p sampling, also called nucleus sampling, is an alternative that restricts
the choice to the smallest set of tokens whose cumulative probability exceeds a
threshold p. Top-k sampling limits the choice to the k most likely tokens.

These parameters trade off consistency against diversity. For factual extraction
or classification you usually want low randomness; for brainstorming or fiction
you want more. Note that even at temperature zero, outputs are not guaranteed to
be byte-for-byte identical across runs.
