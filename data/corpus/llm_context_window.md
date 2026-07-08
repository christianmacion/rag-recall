# LLM Context Window

The context window is the maximum number of tokens a language model can consider
in a single request, counting both the input prompt and the generated output.
If a conversation grows beyond this limit, the earliest content must be dropped
or summarized.

Larger context windows let a model reason over more material at once, such as a
long document or an entire codebase, but processing more tokens increases latency
and cost. Some models offer windows of a million tokens or more.

Two techniques manage long conversations: compaction, which summarizes earlier
turns into a compact block, and context editing, which prunes stale tool results.
Retrieval-augmented generation sidesteps the limit entirely by fetching only the
most relevant passages instead of stuffing everything into the prompt.
