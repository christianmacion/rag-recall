# Python List Comprehensions

A list comprehension is a concise way to build a list in Python. It has the
form `[expression for item in iterable if condition]`. For example,
`[x * x for x in range(5)]` produces `[0, 1, 4, 9, 16]`.

List comprehensions are generally faster than an equivalent `for` loop with
`.append()` because the iteration happens in optimized C code rather than in
the interpreter loop. The optional `if` clause filters items before the
expression is applied.

Dictionary and set comprehensions use the same syntax with braces:
`{k: v for k, v in pairs}` builds a dict, and `{x for x in items}` builds a set.
Overusing nested comprehensions hurts readability, so a plain loop is preferred
once the logic spans more than two clauses.
