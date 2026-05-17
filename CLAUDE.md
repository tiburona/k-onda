# CLAUDE.md — guidance for AI assistants working on K-Onda

## Read this first

Read the README before suggesting edits to the codebase. 

## Raising concerns

If a request seems architecturally wrong, conflicts with the README principles, or has significant tradeoffs, say why before proposing an implementation plan. A one-sentence flag upfront is much less disruptive than mid-implementation course corrections.

## Role of AI in this project

The developer writes her own code except for rote boilerplate. Favor explanation, review, and feedback over generating large amounts of novel code. When code generation is appropriate, keep it small and targeted.

## Local test commands

Pytest is installed in the repository virtual environment. When running tests locally, use:

```bash
.venv/bin/python -m pytest tests
```

Do not use bare `python -m pytest` or `pytest`; the system interpreter may not have the project dependencies installed. If tests fail after running through `.venv/bin/python`, treat them as real test failures rather than a missing-pytest environment problem.

## Architecture overview

See the overview [here](docs/architecture.md).