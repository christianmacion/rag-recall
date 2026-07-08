# Python Generators

A generator is a function that uses the `yield` keyword to produce a sequence of
values lazily, one at a time, instead of building the whole sequence in memory.
Calling a generator function returns a generator object; values are produced
only when you iterate over it.

Generators are memory efficient because they hold only the current value and the
function's local state, not the entire result set. This makes them ideal for
streaming large files or infinite sequences.

A generator expression looks like a list comprehension but uses parentheses:
`(x * x for x in range(1000000))`. It does not allocate the full list. Once a
generator is exhausted it cannot be reused; you must create a new one to iterate
again.
