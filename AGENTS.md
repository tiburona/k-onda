# AGENTS.md

Read `README.md` before suggesting code changes. The numbered principles there are the design constitution of this project.

## Local Test Commands

Pytest is installed in the repository virtual environment. When running tests locally, use:

```bash
.venv/bin/python -m pytest tests
```

Do not use bare `python -m pytest` or `pytest`; the system interpreter may not have the project dependencies installed. If tests fail after running through `.venv/bin/python`, treat them as real test failures rather than a missing-pytest environment problem.
