# Python Decorators

A decorator is a callable that takes a function and returns a new function,
usually to add behavior without modifying the original. The `@decorator` syntax
above a function definition is shorthand for `func = decorator(func)`.

Common uses include logging, timing, caching, access control, and registering
functions. The standard library provides `functools.wraps`, which a decorator
applies to its wrapper so the wrapped function keeps its original name and
docstring.

Decorators can take arguments by adding an extra layer of nesting: an outer
function receives the arguments and returns the actual decorator. `functools.lru_cache`
is a built-in decorator that memoizes a function's return values keyed by its
arguments.
