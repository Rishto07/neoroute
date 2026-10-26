# Neoroute Integration Report

This report reflects the repository state verified from `origin/main` at commit `cbd8354`.

## Merged into main

- M0-M1 foundation: provider attempt tracking, CI, and project setup.
- M2 capability registry: fail-closed capability-aware routing.
- M3 response contract: structural validation of successful upstream responses.
- M4 response corpus: replayable response regression corpus.
- M5 reliability evaluator: deterministic corpus evaluation and baseline comparison.
- M6-related evaluator hardening from `fix/evaluator-baseline-errors`.
- Developer quickstart documentation.
- Product validation research report.
- QA regression tests, including invalid UTF-8 corpus fixture handling.
- Reliability evaluation dashboard MVP at `/` and `/ui/`.

## Outstanding work

- `feat/m6-regression-attribution` is not merged into `main`.
  - It changes `tests/corpus_eval.py`, `tests/test_corpus_eval.py`, and the committed baseline.
  - It adds regression-attribution reporting for corpus evaluation.
  - It should not be described as released functionality until merged and reviewed.

## Verified commands

```powershell
python -m pytest -q
python tests/corpus_eval.py
python -m ruff check .
```

Latest verified results on `origin/main`:

- Tests: 363 passed, 1 skipped.
- Evaluator: 50/50 cases passed.
- Ruff: 3 pre-existing import-sorting findings.

## Blockers and notes

- The dashboard is local-only and has no authentication, rate limiting, or hardened SSRF controls.
- `ruff` reports pre-existing import-sorting findings; CI does not run ruff.
- Generated reports are written under `reports/`, which is git-ignored.
- The M6 regression-attribution branch changes the committed baseline and needs separate review before merge.
- No live provider calls were used for this documentation verification.