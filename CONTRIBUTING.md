# Contributing to Neoroute

Thanks for helping improve Neoroute. This guide covers the local workflow used by the project.

## Prerequisites

- Python 3.11 or newer (CI tests 3.11, 3.12, 3.13, and 3.14).
- Git.
- No real provider API key is required to run the test suite.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` locally. Never commit `.env` or real keys. `.env.example` uses placeholder values such as `replace-me`.

## Run the server

```powershell
uvicorn neoroute.main:app --reload
```

- Health check: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`
- Endpoint: `POST /v1/chat/completions`

Use `model: "auto"` for configured provider order, or a configured provider name to prefer that provider.

## Tests and lint

```powershell
python -m pytest -q
python -m ruff check .
```

The test suite runs without provider credentials; it uses mocked transports. CI (`.github/workflows/ci.yml`) installs the project and runs `python -m pytest -q` on Python 3.11-3.14.

## Reliability evaluator

The evaluator replays the committed response corpus through the real verifier and gateway in-process. No provider is contacted.

```powershell
python tests/corpus_eval.py
```

Exit codes:

- `0` - all cases passed, no regressions, baseline present or allowed
- `1` - case failures, regressions, or strict baseline drift
- `2` - configuration, I/O, or invalid-baseline error

Useful options: `--strict-baseline`, `--allow-missing-baseline`, `--write-baseline`, `--json-out -`.

## When checks fail

- Read the failing test or lint message before editing.
- Fix the code or documentation that caused the failure, then rerun the same command.
- For `ruff`, address findings in files you changed. Do not reformat unrelated files.
- If a command cannot be run in your environment, state that clearly instead of claiming it passed.

## Branch and pull request workflow

1. Create a branch from `main`: `git checkout -b feat/your-change`
2. Make focused changes.
3. Run tests (and lint) before committing.
4. Commit with a clear message.
5. Push the branch: `git push -u origin feat/your-change`
6. Open a pull request against `main`.
7. Do not merge your own PR; wait for review.

## Safety rules

- Do not commit `.env`, real API keys, or provider secrets.
- Do not edit the evaluator implementation, test-corpus fixtures, or committed baseline unless the change is specifically intended and reviewed.
- Before committing, run `git status` and `git diff --check`, and review `git diff`.
- Do not stage `.env`, credentials, corpus fixtures, or baseline changes unless that change is deliberate and reviewed.
- Prefer documentation-only changes for onboarding improvements.
