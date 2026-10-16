# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- This is a student assignment: only `src/assignment/agent/` (plus artifacts and `prompts.py`) is submitted and graded. Never edit `tests/`, `tasks/`, `chess_app/`, or the "Do not modify" blocks (rules in `ASSIGNMENT.md`). Repo-side tests go in `tests_sandbox/`.
- Sandboxes run on Modal by default, or on local Docker with `SANDBOX=docker` (environment or `.env`). Backends live in `src/assignment/sandbox/`; usage is in `README.md`. `make test` runs only the offline tests; `make test-docker` (free) and `make test-modal` (billable) start real sandboxes.
- `make setup` initializes the `chess_app` submodule. Without it, testbed builds and the chess tests fail.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
