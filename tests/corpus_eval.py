"""Deterministic reliability evaluation for the committed response corpus.

The runner replays every M4 fixture through the real verifier and gateway
in-process, summarizes coverage and gateway behavior, compares the result with
a versioned baseline, and emits a byte-stable JSON report plus a terminal
summary. No provider is contacted, no real credential is read, and no semantic
or factual quality is scored.

Usage::

    python tests/corpus_eval.py
    python tests/corpus_eval.py --strict-baseline
    python tests/corpus_eval.py --allow-missing-baseline --write-baseline
    python tests/corpus_eval.py --json-out -

Exit codes:
    0  all cases passed; no regressions; baseline present or explicitly allowed
    1  case failures, regressions, or strict baseline drift
    2  configuration, I/O, or invalid-baseline error
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``python tests/corpus_eval.py`` puts ``tests/`` on sys.path but not the
# repository root. Make the production package importable without requiring an
# editable install or a specific working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx
from corpus_loader import Case, FixtureError, load_cases
from corpus_support import (
    LEAK_MARKER,
    UNEXERCISED_CATEGORIES,
    EnvPatch,
    build_handler,
    build_payload,
    build_settings,
)
from fastapi import HTTPException

from neoroute.attempts import AttemptRecord, ErrorCategory, Outcome
from neoroute.gateway import forward_chat
from neoroute.response_contract import ResponseFailureReason, check_response

SCHEMA_VERSION = 2
RUNNER_NAME = "neoroute-response-eval"
RUNNER_VERSION = 1

REPO_ROOT = _REPO_ROOT
DEFAULT_CORPUS = REPO_ROOT / "tests" / "fixtures" / "response_corpus"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "fixtures" / "response_corpus_baseline.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "response_eval.json"

EXIT_OK = 0
EXIT_EVALUATION_FAILED = 1
EXIT_CONFIGURATION_ERROR = 2

KINDS = frozenset({"verify", "gateway"})
FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class EvalError(Exception):
    """A configuration, I/O, or baseline problem; maps to exit code 2."""


@dataclass(frozen=True)
class CaseResult:
    """Outcome of replaying one corpus fixture."""

    id: str
    kind: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_report(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class Baseline:
    """Validated baseline contents."""

    fingerprint: str
    cases: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class BaselineComparison:
    """Deterministic diff between a baseline and the current evaluation."""

    present: bool
    fingerprint_match: bool | None
    new_cases: tuple[str, ...]
    removed_cases: tuple[str, ...]
    # A case whose baseline ``kind`` no longer matches the corpus is baseline
    # drift, not a pass/fail regression. It is reported and, like added or
    # removed ids, only fails the run under ``--strict-baseline``.
    kind_changes: tuple[str, ...]
    newly_failing: tuple[str, ...]
    newly_passing: tuple[str, ...]
    signature_changes: tuple[str, ...]
    signature_details: tuple[dict[str, Any], ...]
    category_deltas: dict[str, dict[str, int]]

    def as_report(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "fingerprint_match": self.fingerprint_match,
            "new_cases": list(self.new_cases),
            "removed_cases": list(self.removed_cases),
            "kind_changes": list(self.kind_changes),
            "newly_failing": list(self.newly_failing),
            "newly_passing": list(self.newly_passing),
            "signature_changes": list(self.signature_changes),
            "signature_details": [dict(item) for item in self.signature_details],
            "category_deltas": self.category_deltas,
        }


# --- Evaluation -----------------------------------------------------------


def _sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evaluate_verify(case: Case) -> CaseResult:
    expectation = case.verify
    assert expectation is not None
    check = check_response(
        case.body, expects_json_object=case.expects_json_object
    )
    expected = {
        "ok": expectation.ok,
        "reason": expectation.reason.value if expectation.reason else None,
    }
    actual = {
        "ok": check.ok,
        "reason": check.reason.value if check.reason else None,
    }
    failures: list[str] = []
    if actual != expected:
        failures.append(
            f"expected ok={expected['ok']!r} reason={expected['reason']!r}, "
            f"got ok={actual['ok']!r} reason={actual['reason']!r}"
        )
    return CaseResult(
        id=case.id,
        kind=case.kind,
        expected=expected,
        actual=actual,
        failures=tuple(failures),
    )


def _expected_attempt(attempt) -> dict[str, Any]:
    return {
        "outcome": attempt.outcome.value,
        "status_code": attempt.status_code,
        "error_category": (
            attempt.error_category.value if attempt.error_category else None
        ),
        "fallback_attempted": attempt.fallback_attempted,
    }


def _actual_attempt(record: AttemptRecord) -> dict[str, Any]:
    return {
        "attempt": record.attempt,
        "provider": record.provider,
        "outcome": record.outcome.value,
        "status_code": record.status_code,
        "error_category": (
            record.error_category.value if record.error_category else None
        ),
        "fallback_attempted": record.fallback_attempted,
    }


def _gateway_expected(case: Case) -> dict[str, Any]:
    expectation = case.gateway
    assert expectation is not None
    expected: dict[str, Any] = {
        "outcome": expectation.outcome,
        "status_code": expectation.status_code,
    }
    if expectation.outcome == "success":
        expected["provider_index"] = expectation.provider_index
        expected["content_type"] = expectation.content_type
        expected["body_sha256"] = _sha256_hex(expectation.body or b"")
    expected["attempts"] = [
        _expected_attempt(attempt) for attempt in expectation.attempts
    ]
    return expected


async def evaluate_gateway(case: Case) -> CaseResult:
    expectation = case.gateway
    assert expectation is not None
    calls: list[str] = []
    records: list[AttemptRecord] = []
    transport = httpx.MockTransport(build_handler(case, calls))

    with EnvPatch() as patcher:
        settings = build_settings(case, patcher)
        async with httpx.AsyncClient(transport=transport) as client:
            try:
                result, _ = await forward_chat(
                    build_payload(case),
                    settings,
                    client=client,
                    attempt_log=records,
                )
            except HTTPException as exc:
                detail = exc.detail
                actual: dict[str, Any] = {
                    "outcome": "error",
                    "status_code": exc.status_code,
                    "provider": (
                        detail.get("provider")
                        if isinstance(detail, dict)
                        else None
                    ),
                    "detail": detail,
                    "http_calls": list(calls),
                    "attempts": [_actual_attempt(r) for r in records],
                }
            else:
                actual = {
                    "outcome": "success",
                    "provider": result.provider_name,
                    "status_code": result.status_code,
                    "content_type": result.content_type,
                    "body_sha256": _sha256_hex(result.body),
                    "http_calls": list(calls),
                    "attempts": [_actual_attempt(r) for r in records],
                }

    expected = _gateway_expected(case)
    failures = _gateway_failures(case, expected, actual)
    return CaseResult(
        id=case.id,
        kind=case.kind,
        expected=expected,
        actual=actual,
        failures=tuple(failures),
    )


def _gateway_failures(
    case: Case, expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    if actual["outcome"] != expected["outcome"]:
        failures.append(
            f"terminal outcome expected {expected['outcome']!r} "
            f"got {actual['outcome']!r}"
        )
    if actual["status_code"] != expected["status_code"]:
        failures.append(
            f"terminal status_code expected {expected['status_code']!r} "
            f"got {actual['status_code']!r}"
        )

    if expected["outcome"] == "success":
        # Success-only fields are absent from a terminal-error result, so
        # compare them only when the run actually succeeded. The outcome
        # mismatch above already records the regression in every other case.
        if actual["outcome"] == "success":
            provider_index = expected["provider_index"]
            expected_provider = case.providers[provider_index].name
            if actual["provider"] != expected_provider:
                failures.append(
                    f"provider expected {expected_provider!r} "
                    f"got {actual['provider']!r}"
                )
            if actual["content_type"] != expected["content_type"]:
                failures.append(
                    f"content_type expected {expected['content_type']!r} "
                    f"got {actual['content_type']!r}"
                )
            if actual["body_sha256"] != expected["body_sha256"]:
                failures.append(
                    "body_sha256 does not match the expected response"
                )
    elif LEAK_MARKER in str(actual.get("detail")):
        failures.append("error detail leaked the fixture leak marker")

    expected_attempts = expected["attempts"]
    actual_attempts = actual["attempts"]
    if len(actual_attempts) != len(expected_attempts):
        failures.append(
            f"attempt count expected {len(expected_attempts)} "
            f"got {len(actual_attempts)}"
        )
        return failures

    expected_calls = 0
    for index, (expected_attempt, actual_attempt) in enumerate(
        zip(expected_attempts, actual_attempts, strict=True), start=1
    ):
        provider_name = case.providers[index - 1].name
        if actual_attempt["attempt"] != index:
            failures.append(
                f"attempt {index} recorded position "
                f"{actual_attempt['attempt']!r}"
            )
        if actual_attempt["provider"] != provider_name:
            failures.append(
                f"attempt {index} provider expected {provider_name!r} "
                f"got {actual_attempt['provider']!r}"
            )
        for field in (
            "outcome",
            "status_code",
            "error_category",
            "fallback_attempted",
        ):
            if actual_attempt[field] != expected_attempt[field]:
                failures.append(
                    f"attempt {index} {field} expected "
                    f"{expected_attempt[field]!r} "
                    f"got {actual_attempt[field]!r}"
                )
        if (
            expected_attempt["error_category"]
            != ErrorCategory.MISSING_API_KEY.value
        ):
            expected_calls += 1

    if len(actual["http_calls"]) != expected_calls:
        failures.append(
            f"http call count expected {expected_calls} "
            f"got {len(actual['http_calls'])}"
        )
    return failures


async def evaluate_corpus(cases: Sequence[Case]) -> list[CaseResult]:
    """Replay every case in a stable order."""
    results: list[CaseResult] = []
    for case in sorted(cases, key=lambda item: item.id):
        if case.kind == "verify":
            results.append(evaluate_verify(case))
        else:
            results.append(await evaluate_gateway(case))
    return results


# --- Coverage and gateway summaries ---------------------------------------


def build_coverage(cases: Sequence[Case]) -> dict[str, Any]:
    """Summarize enum coverage declared by the fixtures, like the M4 tests."""
    covered_reasons = sorted(
        {
            case.verify.reason.value
            for case in cases
            if case.kind == "verify"
            and case.verify is not None
            and case.verify.reason is not None
        }
    )
    all_reasons = sorted(reason.value for reason in ResponseFailureReason)
    missing_reasons = sorted(set(all_reasons) - set(covered_reasons))

    covered_categories = sorted(
        {
            attempt.error_category.value
            for case in cases
            if case.gateway is not None
            for attempt in case.gateway.attempts
            if attempt.error_category is not None
        }
    )
    documented_exemptions = {
        category.value: reason
        for category, reason in UNEXERCISED_CATEGORIES.items()
    }
    all_categories = {category.value for category in ErrorCategory}
    missing_categories = sorted(
        all_categories
        - set(covered_categories)
        - set(documented_exemptions)
    )

    covered_outcomes = sorted(
        {
            attempt.outcome.value
            for case in cases
            if case.gateway is not None
            for attempt in case.gateway.attempts
        }
    )
    all_outcomes = sorted(outcome.value for outcome in Outcome)
    missing_outcomes = sorted(set(all_outcomes) - set(covered_outcomes))

    return {
        "response_failure_reasons": {
            "covered": covered_reasons,
            "missing": missing_reasons,
            "complete": not missing_reasons,
        },
        "error_categories": {
            "covered": covered_categories,
            "documented_exemptions": documented_exemptions,
            "missing": missing_categories,
            "complete": not missing_categories,
        },
        "attempt_outcomes": {
            "covered": covered_outcomes,
            "missing": missing_outcomes,
            "complete": not missing_outcomes,
        },
    }


def build_gateway_summary(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Summarize observed gateway terminal results and attempts."""
    gateway_results = [result for result in results if result.kind == "gateway"]
    terminal = Counter(result.actual["outcome"] for result in gateway_results)
    attempts = [
        attempt
        for result in gateway_results
        for attempt in result.actual.get("attempts", [])
    ]
    outcome_counts = Counter(attempt["outcome"] for attempt in attempts)
    fallback_flags = Counter(attempt["fallback_attempted"] for attempt in attempts)

    fallback_success = sum(
        1
        for result in gateway_results
        if result.actual["outcome"] == "success"
        and len(result.actual["attempts"]) > 1
    )
    immediate_success = sum(
        1
        for result in gateway_results
        if result.actual["outcome"] == "success"
        and len(result.actual["attempts"]) == 1
    )

    return {
        "terminal_results": {
            "success": terminal.get("success", 0),
            "error": terminal.get("error", 0),
        },
        "fallback_success": fallback_success,
        "immediate_success": immediate_success,
        "attempts": {
            "total": len(attempts),
            "success": outcome_counts.get("success", 0),
            "failure": outcome_counts.get("failure", 0),
            "timeout": outcome_counts.get("timeout", 0),
        },
        "fallback_attempted": {
            "true": fallback_flags.get(True, 0),
            "false": fallback_flags.get(False, 0),
        },
    }


# --- Corpus fingerprint ---------------------------------------------------


def corpus_fingerprint(cases: Sequence[Case]) -> str:
    """Hash sorted case IDs and canonical parsed JSON, independent of newlines."""
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.id):
        try:
            raw = case.path.read_bytes()
            document = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EvalError(
                f"cannot fingerprint fixture {case.path}: {exc}"
            ) from exc
        canonical = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        digest.update(case.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(canonical).digest())
    return f"sha256:{digest.hexdigest()}"


# --- Baseline validation and comparison -----------------------------------


def _reject_keys(
    value: dict[str, Any], allowed: frozenset[str], label: str, path: Path
) -> None:
    missing = sorted(allowed - set(value))
    if missing:
        raise EvalError(
            f"invalid baseline {path}: {label} is missing key(s): "
            f"{', '.join(missing)}"
        )
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvalError(
            f"invalid baseline {path}: {label} has unknown key(s): "
            f"{', '.join(unknown)}"
        )


def _require_object(value: Any, label: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvalError(
            f"invalid baseline {path}: {label} must be a JSON object"
        )
    return value


def _validate_signature(
    signature: Any, kind: str, label: str, path: Path
) -> None:
    if not isinstance(signature, dict) or not signature:
        raise EvalError(
            f"invalid baseline {path}: {label}.signature must be a "
            "non-empty JSON object"
        )
    if kind == "verify":
        _reject_keys(signature, frozenset({"ok", "reason"}), f"{label}.signature", path)
        if not isinstance(signature["ok"], bool):
            raise EvalError(
                f"invalid baseline {path}: {label}.signature.ok must be a boolean"
            )
        reason = signature["reason"]
        if reason is not None and not isinstance(reason, str):
            raise EvalError(
                f"invalid baseline {path}: {label}.signature.reason must be "
                "a string or null"
            )
    else:
        _reject_keys(
            signature,
            frozenset({"outcome", "status_code", "attempts"}),
            f"{label}.signature",
            path,
        )
        outcome = signature["outcome"]
        if outcome not in ("success", "error"):
            raise EvalError(
                f"invalid baseline {path}: {label}.signature.outcome must be "
                "'success' or 'error'"
            )
        status_code = signature["status_code"]
        if status_code is not None and not isinstance(status_code, int):
            raise EvalError(
                f"invalid baseline {path}: {label}.signature.status_code must be "
                "an integer or null"
            )
        attempts = signature["attempts"]
        if not isinstance(attempts, list):
            raise EvalError(
                f"invalid baseline {path}: {label}.signature.attempts must be "
                "an array"
            )
        for index, attempt in enumerate(attempts):
            item_label = f"{label}.signature.attempts[{index}]"
            if not isinstance(attempt, dict):
                raise EvalError(
                    f"invalid baseline {path}: {item_label} must be a JSON object"
                )
            _reject_keys(
                attempt, frozenset({"outcome", "error_category"}), item_label, path
            )
            if attempt["outcome"] not in ("success", "failure", "timeout"):
                raise EvalError(
                    f"invalid baseline {path}: {item_label}.outcome must be one "
                    "of ['failure', 'success', 'timeout']"
                )
            category = attempt["error_category"]
            if category is not None and not isinstance(category, str):
                raise EvalError(
                    f"invalid baseline {path}: {item_label}.error_category must "
                    "be a string or null"
                )


def load_baseline(path: Path) -> Baseline | None:
    """Load and strictly validate a baseline, or return None when absent."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalError(f"cannot read baseline {path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(
            f"invalid baseline {path}: not valid JSON ({exc.msg})"
        ) from exc
    except ValueError as exc:
        raise EvalError(
            f"invalid baseline {path}: not valid JSON ({exc})"
        ) from exc

    root = _require_object(document, "root", path)
    _reject_keys(
        root, frozenset({"schema_version", "corpus", "cases"}), "root", path
    )

    version = root["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise EvalError(
            f"invalid baseline {path}: schema_version must be an integer"
        )
    if version != SCHEMA_VERSION:
        raise EvalError(
            f"invalid baseline {path}: unsupported schema_version {version!r}"
        )

    corpus = _require_object(root["corpus"], "corpus", path)
    _reject_keys(
        corpus, frozenset({"case_count", "fingerprint"}), "corpus", path
    )
    case_count = corpus["case_count"]
    if (
        not isinstance(case_count, int)
        or isinstance(case_count, bool)
        or case_count < 0
    ):
        raise EvalError(
            f"invalid baseline {path}: corpus.case_count must be a "
            "non-negative integer"
        )
    fingerprint = corpus["fingerprint"]
    if not isinstance(fingerprint, str) or not FINGERPRINT_PATTERN.fullmatch(
        fingerprint
    ):
        raise EvalError(
            f"invalid baseline {path}: corpus.fingerprint must be "
            "'sha256:<64 lowercase hex>'"
        )

    raw_cases = root["cases"]
    if not isinstance(raw_cases, list):
        raise EvalError(f"invalid baseline {path}: cases must be an array")

    cases: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        label = f"cases[{index}]"
        item = _require_object(raw_case, label, path)
        _reject_keys(
            item, frozenset({"id", "kind", "passed", "signature"}), label, path
        )
        case_id = item["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvalError(
                f"invalid baseline {path}: {label}.id must be a non-empty "
                "string"
            )
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in KINDS:
            raise EvalError(
                f"invalid baseline {path}: {label}.kind must be one of "
                f"{sorted(KINDS)}"
            )
        passed = item["passed"]
        if not isinstance(passed, bool):
            raise EvalError(
                f"invalid baseline {path}: {label}.passed must be a boolean"
            )
        signature = item["signature"]
        _validate_signature(signature, kind, label, path)
        if case_id in cases:
            raise EvalError(
                f"invalid baseline {path}: duplicate case id {case_id!r}"
            )
        cases[case_id] = {
            "kind": kind,
            "passed": passed,
            "signature": signature,
        }

    if len(cases) != case_count:
        raise EvalError(
            f"invalid baseline {path}: corpus.case_count is {case_count} "
            f"but cases contains {len(cases)} entries"
        )

    return Baseline(fingerprint=fingerprint, cases=cases)


def _case_signature(result: CaseResult) -> dict[str, Any]:
    """Compact, payload-free signature of an evaluated case's actual result."""
    if result.kind == "verify":
        return {
            "ok": result.actual["ok"],
            "reason": result.actual["reason"],
        }
    return {
        "outcome": result.actual["outcome"],
        "status_code": result.actual["status_code"],
        "attempts": [
            {
                "outcome": attempt["outcome"],
                "error_category": attempt["error_category"],
            }
            for attempt in result.actual.get("attempts", [])
        ],
    }


def _changed_signature_fields(
    expected: dict[str, Any] | None, actual: dict[str, Any] | None
) -> list[str]:
    if expected is None or actual is None:
        return sorted(set(expected or {}) | set(actual or {}))
    return sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )


def _count_verify_reasons(signature: dict[str, Any] | None) -> dict[str, int]:
    if not signature:
        return {}
    reason = signature.get("reason")
    return {reason: 1} if reason is not None else {}


def _count_gateway_categories(signature: dict[str, Any] | None) -> dict[str, int]:
    if not signature:
        return {}
    counts: dict[str, int] = {}
    for attempt in signature.get("attempts", []):
        category = attempt.get("error_category")
        if category is not None:
            counts[category] = counts.get(category, 0) + 1
    return counts


def _count_gateway_outcomes(signature: dict[str, Any] | None) -> dict[str, int]:
    if not signature:
        return {}
    counts: dict[str, int] = {}
    for attempt in signature.get("attempts", []):
        outcome = attempt.get("outcome")
        if outcome is not None:
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _subtract_counts(
    left: dict[str, int], right: dict[str, int]
) -> dict[str, int]:
    keys = sorted(set(left) | set(right))
    result: dict[str, int] = {}
    for key in keys:
        delta = left.get(key, 0) - right.get(key, 0)
        if delta:
            result[key] = delta
    return result


def compare_baseline(
    baseline: Baseline | None,
    fingerprint: str,
    results: Sequence[CaseResult],
) -> BaselineComparison:
    if baseline is None:
        return BaselineComparison(
            present=False,
            fingerprint_match=None,
            new_cases=(),
            removed_cases=(),
            kind_changes=(),
            newly_failing=(),
            newly_passing=(),
            signature_changes=(),
            signature_details=(),
            category_deltas={
                "response_failure_reasons": {},
                "error_categories": {},
                "attempt_outcomes": {},
            },
        )

    current = {result.id: result for result in results}
    baseline_ids = set(baseline.cases)
    current_ids = set(current)

    kind_changes = sorted(
        case_id
        for case_id, entry in baseline.cases.items()
        if case_id in current and entry["kind"] != current[case_id].kind
    )
    newly_failing = sorted(
        case_id
        for case_id, entry in baseline.cases.items()
        if entry["passed"] and case_id in current and not current[case_id].passed
    )
    newly_passing = sorted(
        case_id
        for case_id, entry in baseline.cases.items()
        if not entry["passed"]
        and case_id in current
        and current[case_id].passed
    )

    signature_changes: list[str] = []
    signature_details: list[dict[str, Any]] = []
    baseline_reason_counts: dict[str, int] = {}
    current_reason_counts: dict[str, int] = {}
    baseline_category_counts: dict[str, int] = {}
    current_category_counts: dict[str, int] = {}
    baseline_outcome_counts: dict[str, int] = {}
    current_outcome_counts: dict[str, int] = {}

    for case_id, entry in baseline.cases.items():
        if case_id not in current:
            continue
        result = current[case_id]
        if entry["kind"] != result.kind:
            continue
        baseline_signature = entry.get("signature")
        current_signature = _case_signature(result)
        if baseline_signature != current_signature:
            signature_changes.append(case_id)
            signature_details.append(
                {
                    "id": case_id,
                    "expected": baseline_signature,
                    "actual": current_signature,
                    "changed_fields": _changed_signature_fields(
                        baseline_signature, current_signature
                    ),
                }
            )
        if result.kind == "verify":
            for key, value in _count_verify_reasons(baseline_signature).items():
                baseline_reason_counts[key] = (
                    baseline_reason_counts.get(key, 0) + value
                )
            for key, value in _count_verify_reasons(current_signature).items():
                current_reason_counts[key] = (
                    current_reason_counts.get(key, 0) + value
                )
        else:
            for key, value in _count_gateway_categories(baseline_signature).items():
                baseline_category_counts[key] = (
                    baseline_category_counts.get(key, 0) + value
                )
            for key, value in _count_gateway_categories(current_signature).items():
                current_category_counts[key] = (
                    current_category_counts.get(key, 0) + value
                )
            for key, value in _count_gateway_outcomes(baseline_signature).items():
                baseline_outcome_counts[key] = (
                    baseline_outcome_counts.get(key, 0) + value
                )
            for key, value in _count_gateway_outcomes(current_signature).items():
                current_outcome_counts[key] = (
                    current_outcome_counts.get(key, 0) + value
                )

    category_deltas = {
        "response_failure_reasons": _subtract_counts(
            current_reason_counts, baseline_reason_counts
        ),
        "error_categories": _subtract_counts(
            current_category_counts, baseline_category_counts
        ),
        "attempt_outcomes": _subtract_counts(
            current_outcome_counts, baseline_outcome_counts
        ),
    }

    return BaselineComparison(
        present=True,
        fingerprint_match=baseline.fingerprint == fingerprint,
        new_cases=tuple(sorted(current_ids - baseline_ids)),
        removed_cases=tuple(sorted(baseline_ids - current_ids)),
        kind_changes=tuple(kind_changes),
        newly_failing=tuple(newly_failing),
        newly_passing=tuple(newly_passing),
        signature_changes=tuple(sorted(signature_changes)),
        signature_details=tuple(signature_details),
        category_deltas=category_deltas,
    )


# --- Report and baseline documents ----------------------------------------


def build_report(
    cases: Sequence[Case],
    results: Sequence[CaseResult],
    fingerprint: str,
    coverage: dict[str, Any],
    gateway: dict[str, Any],
    comparison: BaselineComparison,
    exit_code: int,
) -> dict[str, Any]:
    passed = sum(1 for result in results if result.passed)
    verify_count = sum(1 for case in cases if case.kind == "verify")
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": {"name": RUNNER_NAME, "version": RUNNER_VERSION},
        "corpus": {
            "case_count": len(cases),
            "verify_count": verify_count,
            "gateway_count": len(cases) - verify_count,
            "fingerprint": fingerprint,
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "cases": [result.as_report() for result in results],
        "coverage": coverage,
        "gateway": gateway,
        "baseline": comparison.as_report(),
        "result": {
            "ok": exit_code == EXIT_OK,
            "exit_code": exit_code,
            "regressions": bool(comparison.newly_failing),
        },
    }


def build_baseline_document(
    cases: Sequence[Case],
    results: Sequence[CaseResult],
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus": {
            "case_count": len(cases),
            "fingerprint": fingerprint,
        },
        "cases": [
            {
                "id": result.id,
                "kind": result.kind,
                "passed": result.passed,
                "signature": _case_signature(result),
            }
            for result in sorted(results, key=lambda item: item.id)
        ],
    }


def determine_exit_code(
    results: Sequence[CaseResult],
    comparison: BaselineComparison,
    *,
    strict_baseline: bool,
) -> int:
    if any(not result.passed for result in results):
        return EXIT_EVALUATION_FAILED
    if comparison.newly_failing:
        return EXIT_EVALUATION_FAILED
    if strict_baseline and (
        comparison.new_cases
        or comparison.removed_cases
        or comparison.kind_changes
        or comparison.signature_changes
    ):
        return EXIT_EVALUATION_FAILED
    return EXIT_OK


def render_json(document: dict[str, Any]) -> str:
    """Render stable, human-readable JSON with a trailing newline."""
    return json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write JSON through a temp file and atomically replace the target."""
    text = render_json(document)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise EvalError(f"cannot write {path}: {exc}") from exc


# --- Terminal summary -----------------------------------------------------


def render_terminal_summary(
    report: dict[str, Any],
    *,
    json_location: str,
    write_refused: bool,
) -> str:
    corpus = report["corpus"]
    summary = report["summary"]
    coverage = report["coverage"]
    gateway = report["gateway"]
    baseline = report["baseline"]

    reasons = coverage["response_failure_reasons"]
    reason_total = len(reasons["covered"]) + len(reasons["missing"])
    categories = coverage["error_categories"]
    category_total = (
        len(categories["covered"])
        + len(categories["missing"])
        + len(categories["documented_exemptions"])
    )
    outcomes = coverage["attempt_outcomes"]
    outcome_total = len(outcomes["covered"]) + len(outcomes["missing"])

    lines = [
        "Neoroute response corpus evaluation",
        (
            f"corpus:   {corpus['case_count']} cases "
            f"({corpus['verify_count']} verify, "
            f"{corpus['gateway_count']} gateway)"
        ),
        f"cases:    PASS {summary['passed']} / FAIL {summary['failed']}",
        (
            f"reasons:  {len(reasons['covered'])}/{reason_total} "
            "ResponseFailureReason coverage"
        ),
    ]

    exemption_count = len(categories["documented_exemptions"])
    exemption_names = ", ".join(sorted(categories["documented_exemptions"]))
    category_line = (
        f"errors:   {len(categories['covered'])}/{category_total} "
        "ErrorCategory coverage"
    )
    if exemption_count:
        category_line += (
            f", {exemption_count} documented exemption(s) ({exemption_names})"
        )
    lines.append(category_line)
    lines.append(
        f"outcomes: {len(outcomes['covered'])}/{outcome_total} "
        "attempt outcomes"
    )
    lines.append(
        f"gateway:  {gateway['terminal_results']['success']} success / "
        f"{gateway['terminal_results']['error']} error; "
        f"fallback {gateway['fallback_success']}, "
        f"immediate {gateway['immediate_success']}"
    )
    attempts = gateway["attempts"]
    lines.append(
        f"attempts: {attempts['total']} ({attempts['success']} success, "
        f"{attempts['failure']} failure, {attempts['timeout']} timeout)"
    )

    if not baseline["present"]:
        lines.append("baseline: missing (allowed), no comparison")
    else:
        fingerprint_status = (
            "match" if baseline["fingerprint_match"] else "mismatch"
        )
        lines.append(
            "baseline: "
            f"fingerprint {fingerprint_status}, "
            f"new {len(baseline['new_cases'])}, "
            f"removed {len(baseline['removed_cases'])}, "
            f"kind changed {len(baseline['kind_changes'])}, "
            f"newly failing {len(baseline['newly_failing'])}, "
            f"newly passing {len(baseline['newly_passing'])}"
        )

    if write_refused:
        lines.append(
            "baseline write: REFUSED (failing cases; use --force to override)"
        )

    for case in report["cases"]:
        if not case["passed"]:
            detail = "; ".join(case["failures"])
            lines.append(f"FAIL {case['id']}: {detail}")

    lines.append(f"json:     {json_location}")

    if report["result"]["ok"]:
        verdict = "PASS"
    else:
        details: list[str] = []
        if summary["failed"]:
            details.append(f"{summary['failed']} case(s) failed")
        if report["result"]["regressions"]:
            details.append(
                f"{len(baseline['newly_failing'])} regression(s)"
            )
        if not details and (
            baseline["new_cases"]
            or baseline["removed_cases"]
            or baseline["kind_changes"]
        ):
            details.append("strict baseline drift")
        verdict = "FAIL"
        if details:
            verdict += f" ({', '.join(details)})"
    lines.append(f"result:   {verdict}")
    return "\n".join(lines) + "\n"


# --- Output path safety ---------------------------------------------------


def _canonical(path: Path) -> str:
    """Return a best-effort canonical key for a path.

    ``resolve`` follows existing symlinks and normalizes ``.``/``..``;
    ``normcase`` folds Windows case differences so two spellings of the same
    file compare equal.
    """
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = Path(os.path.abspath(path))
    return os.path.normcase(os.path.normpath(str(resolved)))


def _same_or_nested(path: Path, directory: Path) -> bool:
    """True when ``path`` is ``directory`` or lives underneath it."""
    path_key = _canonical(path)
    directory_key = _canonical(directory)
    return path_key == directory_key or path_key.startswith(
        directory_key + os.sep
    )


def _validate_destination(path: Path, label: str) -> None:
    """Reject a destination that is a directory or has a file as an ancestor."""
    try:
        if path.exists() and path.is_dir():
            raise EvalError(f"{label} is a directory: {path}")
        parent = path.parent
        while not parent.exists():
            if parent.parent == parent:
                return
            parent = parent.parent
        if not parent.is_dir():
            raise EvalError(f"{label} parent is not a directory: {parent}")
    except OSError as exc:
        raise EvalError(f"invalid {label} path {path}: {exc}") from exc


def validate_destinations(
    corpus: Path,
    baseline: Path,
    report: Path | None,
    *,
    write_baseline: bool,
) -> None:
    """Reject outputs that could overwrite the corpus, baseline, or each other.

    The corpus and baseline are versioned source artifacts, so a mistyped
    ``--json-out`` must fail before any file is touched. ``report`` is ``None``
    for ``--json-out -`` (stdout).
    """
    if report is not None:
        if _canonical(report) == _canonical(baseline):
            raise EvalError(
                "--json-out must not be the same file as --baseline"
            )
        if _same_or_nested(report, corpus):
            raise EvalError(
                f"--json-out must not be inside the corpus directory: {report}"
            )
    if _same_or_nested(baseline, corpus):
        raise EvalError(
            f"--baseline must not be inside the corpus directory: {baseline}"
        )
    if report is not None:
        _validate_destination(report, "--json-out")
    if write_baseline:
        _validate_destination(baseline, "--baseline")


# --- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus_eval.py",
        description=(
            "Replay the Neoroute response corpus deterministically and "
            "compare it with a versioned baseline."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="corpus fixture directory",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="versioned baseline file",
    )
    parser.add_argument(
        "--json-out",
        default=str(DEFAULT_REPORT),
        help="JSON report path, or '-' for stdout",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="evaluate without a baseline instead of failing",
    )
    parser.add_argument(
        "--strict-baseline",
        action="store_true",
        help="treat new or removed case ids and baseline kind changes as failures",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="atomically write the evaluated result to --baseline",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow --write-baseline even when cases fail",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    report_path = None if args.json_out == "-" else Path(args.json_out)
    validate_destinations(
        args.corpus,
        args.baseline,
        report_path,
        write_baseline=args.write_baseline,
    )

    try:
        cases = load_cases(args.corpus)
    except FixtureError as exc:
        raise EvalError(str(exc)) from exc

    baseline = load_baseline(args.baseline)
    if baseline is None and not args.allow_missing_baseline:
        raise EvalError(
            f"baseline not found: {args.baseline} "
            "(use --allow-missing-baseline to evaluate without one)"
        )

    results = asyncio.run(evaluate_corpus(cases))
    fingerprint = corpus_fingerprint(cases)

    comparison = compare_baseline(baseline, fingerprint, results)
    coverage = build_coverage(cases)
    gateway = build_gateway_summary(results)
    exit_code = determine_exit_code(
        results, comparison, strict_baseline=args.strict_baseline
    )
    report = build_report(
        cases,
        results,
        fingerprint,
        coverage,
        gateway,
        comparison,
        exit_code,
    )

    write_refused = False
    if args.write_baseline:
        failed = [result.id for result in results if not result.passed]
        if failed and not args.force:
            write_refused = True
            print(
                "refusing to write baseline: "
                f"{len(failed)} failing case(s); use --force to override",
                file=sys.stderr,
            )

    # The report and baseline are not written transactionally. The report is
    # written first so an output failure can never mutate the baseline; a
    # baseline write failure after a successful report leaves a fresh report
    # beside an unchanged baseline and exits 2.
    json_location = args.json_out
    if report_path is None:
        sys.stdout.write(render_json(report))
        sys.stdout.flush()
        summary_stream = sys.stderr
    else:
        write_json_atomic(report_path, report)
        summary_stream = sys.stdout

    if args.write_baseline and not write_refused:
        write_json_atomic(
            args.baseline,
            build_baseline_document(cases, results, fingerprint),
        )
        print(f"wrote baseline: {args.baseline}", file=sys.stderr)

    summary = render_terminal_summary(
        report,
        json_location=json_location,
        write_refused=write_refused,
    )
    summary_stream.write(summary)
    summary_stream.flush()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
