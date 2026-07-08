# Python Virtual Environments

A virtual environment is an isolated directory containing a specific Python
interpreter and its own set of installed packages. It prevents dependency
conflicts between projects that need different package versions.

The built-in `venv` module creates one with `python -m venv .venv`. You activate
it with `source .venv/bin/activate` on macOS or Linux, and `pip install` then
installs packages into that environment only, not system-wide.

A `requirements.txt` file lists a project's dependencies, one per line, often
pinned to exact versions. Running `pip install -r requirements.txt` reproduces
the environment. Deactivating with the `deactivate` command returns you to the
system interpreter.
