"""Deterministic acceptance tests for the M5 reliability evaluation runner.

Every test uses ``tmp_path`` for reports and baselines. The committed corpus
and baseline are treated as read-only source artifacts.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import corpus_eval
import pytest
from corpus_eval import (
    DEFAULT_BASELINE,
    DEFAULT_CORPUS,
    corpus_fingerprint,
    load_baseline,
    main,
)
from corpus_loader import Case, load_cases
from corpus_support import LEAK_MARKER
from fastapi import HTTPException

FIXTURES = DEFAULT_CORPUS
REAL_BASELINE = DEFAULT_BASELINE


# --- Helpers --------------------------------------------------------------


def _copy_corpus(
    tmp_path: Path, *, line_ending: str | None = None, name: str = "corpus"
) -> Path:
    destination = tmp_path / name
    destination.mkdir()
    for source in sorted(FIXTURES.glob("*.json")):
        raw = source.read_bytes()
        if line_ending == "lf":
            raw = raw.replace(b"\r\n", b"\n")
        elif line_ending == "crlf":
            raw = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        (destination / source.name).write_bytes(raw)
    return destination


def _run(
    tmp_path: Path,
    *,
    corpus: Path = DEFAULT_CORPUS,
    baseline: Path = REAL_BASELINE,
    extra: tuple[str, ...] = (),
    report_name: str = "report.json",
) -> tuple[int, Path]:
    report = tmp_path / report_name
    argv = [
        "--corpus",
        str(corpus),
        "--baseline",
        str(baseline),
        "--json-out",
        str(report),
        *extra,
    ]
    return main(argv), report


def _read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_document() -> dict:
    return json.loads(REAL_BASELINE.read_text(encoding="utf-8"))


def _case(report: dict, case_id: str) -> dict:
    return next(case for case in report["cases"] if case["id"] == case_id)


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _all_keys(item)
    return keys


def _write_failing_verify(corpus: Path) -> None:
    path = corpus / "valid.minimal.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["expect"] = {"ok": False, "reason": "missing_choices"}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _write_extra_verify(corpus: Path, case_id: str = "valid.eval_added") -> None:
    body = '{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
    document = {
        "id": case_id,
        "kind": "verify",
        "description": "Synthetic case added by the evaluator tests.",
        "expects_json_object": False,
        "body": body,
        "expect": {"ok": True, "reason": None},
    }
    (corpus / f"{case_id}.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.glob("*.json"))
    }


# --- Corpus results and summaries ----------------------------------------


def test_real_corpus_matches_versioned_baseline(tmp_path):
    exit_code, report_path = _run(tmp_path)

    assert exit_code == 0
    report = _read_report(report_path)
    assert report["schema_version"] == 2
    assert report["runner"] == {
        "name": "neoroute-response-eval",
        "version": 1,
    }
    assert report["corpus"]["case_count"] == 50
    assert report["corpus"]["verify_count"] == 36
    assert report["corpus"]["gateway_count"] == 14
    assert report["summary"] == {"total": 50, "passed": 50, "failed": 0}
    assert all(case["passed"] for case in report["cases"])
    assert report["baseline"] == {
        "present": True,
        "fingerprint_match": True,
        "new_cases": [],
        "removed_cases": [],
        "kind_changes": [],
        "newly_failing": [],
        "newly_passing": [],
        "signature_changes": [],
        "signature_details": [],
        "category_deltas": {
            "response_failure_reasons": {},
            "error_categories": {},
            "attempt_outcomes": {},
        },
    }
    assert report["result"] == {
        "ok": True,
        "exit_code": 0,
        "regressions": False,
    }


def test_coverage_matches_the_committed_corpus(tmp_path):
    exit_code, report_path = _run(tmp_path)

    assert exit_code == 0
    coverage = _read_report(report_path)["coverage"]
    assert coverage["response_failure_reasons"] == {
        "covered": [
            "body_not_json",
            "empty_choices",
            "invalid_choice",
            "invalid_choices",
            "invalid_content",
            "invalid_function_call",
            "invalid_json_content",
            "invalid_message",
            "invalid_role",
            "invalid_tool_calls",
            "missing_choices",
            "missing_content",
            "missing_message",
            "top_level_not_object",
        ],
        "missing": [],
        "complete": True,
    }
    assert coverage["error_categories"]["covered"] == [
        "connection_error",
        "invalid_json_content",
        "invalid_response",
        "malformed_response",
        "missing_api_key",
        "provider_client_error",
        "provider_server_error",
        "rate_limited",
        "timeout",
    ]
    assert coverage["error_categories"]["missing"] == []
    assert coverage["error_categories"]["complete"] is True
    assert set(coverage["error_categories"]["documented_exemptions"]) == {
        "unknown"
    }
    assert coverage["attempt_outcomes"] == {
        "covered": ["failure", "success", "timeout"],
        "missing": [],
        "complete": True,
    }


def test_gateway_summary_matches_the_committed_corpus(tmp_path):
    exit_code, report_path = _run(tmp_path)

    assert exit_code == 0
    assert _read_report(report_path)["gateway"] == {
        "terminal_results": {"success": 9, "error": 5},
        "fallback_success": 7,
        "immediate_success": 2,
        "attempts": {
            "total": 24,
            "success": 9,
            "failure": 14,
            "timeout": 1,
        },
        "fallback_attempted": {"true": 10, "false": 14},
    }


def test_representative_real_case_results(tmp_path):
    exit_code, report_path = _run(tmp_path)

    assert exit_code == 0
    report = _read_report(report_path)

    minimal = _case(report, "valid.minimal")
    assert minimal["expected"] == {"ok": True, "reason": None}
    assert minimal["actual"] == {"ok": True, "reason": None}

    fallback = _case(report, "gateway.success.rate_limited_then_valid")
    assert fallback["actual"]["outcome"] == "success"
    assert fallback["actual"]["provider"] == "beta"
    assert fallback["actual"]["body_sha256"] == (
        "5f881b1405d3b82bb0198dbecba225e07b0e40ad36d84ca543b4f8933917797e"
    )
    assert [
        (attempt["outcome"], attempt["error_category"])
        for attempt in fallback["actual"]["attempts"]
    ] == [("failure", "rate_limited"), ("success", None)]

    stopped = _case(report, "gateway.error.malformed_then_400_stop")
    assert stopped["actual"]["outcome"] == "error"
    assert stopped["actual"]["status_code"] == 400
    assert stopped["actual"]["attempts"][-1]["error_category"] == (
        "provider_client_error"
    )
    assert stopped["actual"]["detail"] == {
        "message": "Upstream rejected the request; fallback stopped",
        "provider": "beta",
        "upstream_status": 400,
    }


# --- Determinism and sanitization ----------------------------------------


def test_reports_are_byte_identical_across_runs(tmp_path):
    first_exit, first_path = _run(tmp_path, report_name="first.json")
    second_exit, second_path = _run(tmp_path, report_name="second.json")

    assert first_exit == 0
    assert second_exit == 0
    assert first_path.read_bytes() == second_path.read_bytes()


def test_report_has_no_nondeterminism_or_payload_leakage(tmp_path, capsys):
    exit_code, report_path = _run(tmp_path)
    captured = capsys.readouterr()

    assert exit_code == 0
    text = report_path.read_text(encoding="utf-8")
    assert LEAK_MARKER not in text
    assert LEAK_MARKER not in captured.out
    assert LEAK_MARKER not in captured.err
    for forbidden in (
        "generated_at",
        "duration_ms",
        "trace_id",
        "started_at",
        "authorization",
        "Authorization",
        "Bearer ",
        "https://",
        "http://",
    ):
        assert forbidden not in text
    for path in (str(tmp_path), str(FIXTURES)):
        assert path not in text
    assert re.search(r"\b[A-Za-z]:[\\/]", text) is None

    keys = _all_keys(_read_report(report_path))
    assert "body" not in keys
    assert "content" not in keys
    assert "trace_id" not in keys


def test_fingerprint_ignores_line_endings(tmp_path):
    lf_corpus = _copy_corpus(tmp_path, line_ending="lf", name="lf")
    crlf_corpus = _copy_corpus(tmp_path, line_ending="crlf", name="crlf")

    lf_fingerprint = corpus_fingerprint(load_cases(lf_corpus))
    crlf_fingerprint = corpus_fingerprint(load_cases(crlf_corpus))
    real_fingerprint = corpus_fingerprint(load_cases(FIXTURES))

    assert lf_fingerprint == crlf_fingerprint == real_fingerprint
    assert real_fingerprint == (
        "sha256:d17a403e174763588ad24b0b8b30a8e1d9798e0fb7b21bd623d3c5290a3fe2db"
    )


def test_json_out_dash_writes_json_to_stdout_only(tmp_path, capsys):
    exit_code = main(
        [
            "--baseline",
            str(tmp_path / "missing.json"),
            "--allow-missing-baseline",
            "--json-out",
            "-",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["baseline"]["present"] is False
    assert "result:   PASS" in captured.err


# --- Missing and corrupt baselines ---------------------------------------


def test_missing_baseline_exits_2(tmp_path, capsys):
    exit_code, report_path = _run(
        tmp_path, baseline=tmp_path / "does-not-exist.json"
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "baseline not found" in captured.err
    assert "does-not-exist.json" in captured.err
    assert "Traceback" not in captured.err
    assert not report_path.exists()


def test_missing_baseline_is_allowed_explicitly(tmp_path):
    exit_code, report_path = _run(
        tmp_path,
        baseline=tmp_path / "does-not-exist.json",
        extra=("--allow-missing-baseline",),
    )

    assert exit_code == 0
    report = _read_report(report_path)
    assert report["baseline"] == {
        "present": False,
        "fingerprint_match": None,
        "new_cases": [],
        "removed_cases": [],
        "kind_changes": [],
        "newly_failing": [],
        "newly_passing": [],
        "signature_changes": [],
        "signature_details": [],
        "category_deltas": {
            "response_failure_reasons": {},
            "error_categories": {},
            "attempt_outcomes": {},
        },
    }
    assert report["result"] == {
        "ok": True,
        "exit_code": 0,
        "regressions": False,
    }


def test_invalid_json_baseline_exits_2(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{", encoding="utf-8")

    exit_code, report_path = _run(tmp_path, baseline=baseline)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "invalid baseline" in captured.err
    assert "not valid JSON" in captured.err
    assert "Traceback" not in captured.err
    assert not report_path.exists()


def _corrupt_version(document: dict) -> None:
    document["schema_version"] = 3


def _corrupt_duplicate_id(document: dict) -> None:
    document["cases"].append(copy.deepcopy(document["cases"][0]))
    document["corpus"]["case_count"] = len(document["cases"])


def _corrupt_non_boolean(document: dict) -> None:
    document["cases"][0]["passed"] = "yes"


def _corrupt_fingerprint(document: dict) -> None:
    document["corpus"]["fingerprint"] = "sha256:not-a-real-digest"


def _corrupt_case_count(document: dict) -> None:
    document["corpus"]["case_count"] += 1


def _corrupt_unknown_key(document: dict) -> None:
    document["extra"] = True


def _corrupt_missing_key(document: dict) -> None:
    del document["cases"][0]["kind"]


@pytest.mark.parametrize(
    "mutate",
    [
        _corrupt_version,
        _corrupt_duplicate_id,
        _corrupt_non_boolean,
        _corrupt_fingerprint,
        _corrupt_case_count,
        _corrupt_unknown_key,
        _corrupt_missing_key,
    ],
    ids=[
        "version",
        "duplicate-id",
        "non-boolean-passed",
        "malformed-fingerprint",
        "case-count-mismatch",
        "unknown-key",
        "missing-key",
    ],
)
def test_invalid_baseline_exits_2(tmp_path, capsys, mutate):
    document = _baseline_document()
    mutate(document)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(document), encoding="utf-8")

    exit_code, report_path = _run(tmp_path, baseline=baseline)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "invalid baseline" in captured.err
    assert "Traceback" not in captured.err
    assert not report_path.exists()


# --- Baseline drift -------------------------------------------------------


def test_new_case_is_informational_unless_strict(tmp_path):
    corpus = _copy_corpus(tmp_path)
    _write_extra_verify(corpus)

    default_exit, default_report = _run(
        tmp_path, corpus=corpus, report_name="default.json"
    )
    assert default_exit == 0
    report = _read_report(default_report)
    assert report["baseline"]["new_cases"] == ["valid.eval_added"]
    assert report["baseline"]["removed_cases"] == []
    assert report["baseline"]["fingerprint_match"] is False
    assert report["result"]["ok"] is True

    strict_exit, strict_report = _run(
        tmp_path,
        corpus=corpus,
        extra=("--strict-baseline",),
        report_name="strict.json",
    )
    assert strict_exit == 1
    report = _read_report(strict_report)
    assert report["baseline"]["new_cases"] == ["valid.eval_added"]
    assert report["result"] == {
        "ok": False,
        "exit_code": 1,
        "regressions": False,
    }


def test_removed_case_is_informational_unless_strict(tmp_path):
    corpus = _copy_corpus(tmp_path)
    (corpus / "valid.minimal.json").unlink()

    default_exit, default_report = _run(
        tmp_path, corpus=corpus, report_name="default.json"
    )
    assert default_exit == 0
    report = _read_report(default_report)
    assert report["baseline"]["removed_cases"] == ["valid.minimal"]
    assert report["baseline"]["new_cases"] == []
    assert report["result"]["ok"] is True

    strict_exit, strict_report = _run(
        tmp_path,
        corpus=corpus,
        extra=("--strict-baseline",),
        report_name="strict.json",
    )
    assert strict_exit == 1
    report = _read_report(strict_report)
    assert report["baseline"]["removed_cases"] == ["valid.minimal"]
    assert report["result"]["ok"] is False


# --- Failures and regressions --------------------------------------------


def test_case_failure_without_baseline_exits_1(tmp_path):
    corpus = _copy_corpus(tmp_path)
    _write_failing_verify(corpus)

    exit_code, report_path = _run(
        tmp_path,
        corpus=corpus,
        baseline=tmp_path / "missing.json",
        extra=("--allow-missing-baseline",),
    )

    assert exit_code == 1
    report = _read_report(report_path)
    assert report["summary"] == {"total": 50, "passed": 49, "failed": 1}
    failed = _case(report, "valid.minimal")
    assert failed["passed"] is False
    assert failed["actual"] == {"ok": True, "reason": None}
    assert failed["expected"] == {"ok": False, "reason": "missing_choices"}
    assert failed["failures"]
    assert report["result"]["regressions"] is False


def test_newly_failing_case_is_a_regression(tmp_path):
    corpus = _copy_corpus(tmp_path)
    _write_failing_verify(corpus)

    exit_code, report_path = _run(tmp_path, corpus=corpus)

    assert exit_code == 1
    report = _read_report(report_path)
    assert report["baseline"]["newly_failing"] == ["valid.minimal"]
    assert report["result"] == {
        "ok": False,
        "exit_code": 1,
        "regressions": True,
    }


def test_known_failing_baseline_is_not_a_regression(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    _write_failing_verify(corpus)
    baseline = tmp_path / "baseline.json"

    refused_exit, _ = _run(
        tmp_path,
        corpus=corpus,
        baseline=baseline,
        extra=("--allow-missing-baseline", "--write-baseline"),
        report_name="refused.json",
    )
    assert refused_exit == 1
    assert not baseline.exists()
    assert "refusing to write baseline" in capsys.readouterr().err

    forced_exit, _ = _run(
        tmp_path,
        corpus=corpus,
        baseline=baseline,
        extra=(
            "--allow-missing-baseline",
            "--write-baseline",
            "--force",
        ),
        report_name="forced.json",
    )
    assert forced_exit == 1
    assert baseline.exists()
    written = json.loads(baseline.read_text(encoding="utf-8"))
    entry = next(
        case for case in written["cases"] if case["id"] == "valid.minimal"
    )
    assert entry["passed"] is False

    exit_code, report_path = _run(
        tmp_path,
        corpus=corpus,
        baseline=baseline,
        report_name="known-failure.json",
    )
    assert exit_code == 1
    report = _read_report(report_path)
    assert report["summary"]["failed"] == 1
    assert report["baseline"]["newly_failing"] == []
    assert report["baseline"]["newly_passing"] == []
    assert report["result"]["regressions"] is False


def test_newly_passing_case_is_reported(tmp_path):
    document = _baseline_document()
    entry = next(
        case for case in document["cases"] if case["id"] == "valid.minimal"
    )
    entry["passed"] = False
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(document), encoding="utf-8")

    exit_code, report_path = _run(tmp_path, baseline=baseline)

    assert exit_code == 0
    report = _read_report(report_path)
    assert report["summary"]["failed"] == 0
    assert report["baseline"]["newly_passing"] == ["valid.minimal"]
    assert report["baseline"]["newly_failing"] == []
    assert report["result"]["ok"] is True


# --- Baseline writing -----------------------------------------------------


def test_baseline_write_round_trip(tmp_path):
    baseline = tmp_path / "baseline.json"

    write_exit, _ = _run(
        tmp_path,
        baseline=baseline,
        extra=("--allow-missing-baseline", "--write-baseline"),
        report_name="write.json",
    )

    assert write_exit == 0
    assert baseline.exists()
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())

    written = json.loads(baseline.read_text(encoding="utf-8"))
    assert written["schema_version"] == 2
    assert written["corpus"]["case_count"] == 50
    assert len(written["cases"]) == 50
    assert [case["id"] for case in written["cases"]] == sorted(
        case["id"] for case in written["cases"]
    )
    assert all(case["passed"] for case in written["cases"])

    loaded = load_baseline(baseline)
    assert loaded is not None
    assert loaded.fingerprint == corpus_fingerprint(load_cases(FIXTURES))

    reload_exit, report_path = _run(
        tmp_path, baseline=baseline, report_name="reload.json"
    )
    assert reload_exit == 0
    report = _read_report(report_path)
    assert report["baseline"]["fingerprint_match"] is True
    assert report["baseline"]["new_cases"] == []
    assert report["baseline"]["removed_cases"] == []
    assert report["result"]["ok"] is True


# --- Exit-code contract ---------------------------------------------------


def test_exit_code_contract(tmp_path):
    success_exit, _ = _run(tmp_path, report_name="success.json")
    assert success_exit == 0

    corpus = _copy_corpus(tmp_path, name="failing")
    _write_failing_verify(corpus)
    failure_exit, _ = _run(
        tmp_path, corpus=corpus, report_name="failure.json"
    )
    assert failure_exit == 1

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("[]", encoding="utf-8")
    config_exit, _ = _run(
        tmp_path, baseline=corrupt, report_name="config.json"
    )
    assert config_exit == 2


# --- Gateway failure handling --------------------------------------------


def _first_success_gateway() -> Case:
    return next(
        case
        for case in load_cases()
        if case.kind == "gateway"
        and case.gateway is not None
        and case.gateway.outcome == "success"
    )


def test_gateway_success_expected_but_error_actual_is_a_failure():
    case = _first_success_gateway()
    expected = corpus_eval._gateway_expected(case)
    actual = {
        "outcome": "error",
        "status_code": 502,
        "provider": None,
        "detail": {"message": "simulated provider failure"},
        "http_calls": [],
        "attempts": [],
    }

    failures = corpus_eval._gateway_failures(case, expected, actual)

    assert failures
    assert any(
        "terminal outcome expected 'success' got 'error'" in failure
        for failure in failures
    )
    # The error path has no success-only fields, so they must not be read.
    assert not any(
        "content_type" in failure or "body_sha256" in failure
        for failure in failures
    )


def test_gateway_success_to_error_regression_writes_report(
    tmp_path, capsys, monkeypatch
):
    async def fail_forward_chat(*args, **kwargs):
        raise HTTPException(
            status_code=502,
            detail={"message": "simulated provider failure"},
        )

    monkeypatch.setattr(corpus_eval, "forward_chat", fail_forward_chat)
    corpus = _copy_corpus(tmp_path)
    success_case = _first_success_gateway()

    exit_code, report_path = _run(
        tmp_path,
        corpus=corpus,
        baseline=tmp_path / "missing.json",
        extra=("--allow-missing-baseline",),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert report_path.exists()
    assert "Traceback" not in captured.err
    report = _read_report(report_path)
    failed = _case(report, success_case.id)
    assert failed["passed"] is False
    assert any(
        "terminal outcome expected 'success' got 'error'" in failure
        for failure in failed["failures"]
    )
    assert report["result"]["ok"] is False


# --- Baseline kind validation and drift ----------------------------------


@pytest.mark.parametrize(
    "kind",
    [[], {}, None, 3, True, "bogus"],
    ids=["list", "object", "null", "integer", "boolean", "unknown"],
)
def test_invalid_baseline_kind_exits_2(tmp_path, capsys, kind):
    document = _baseline_document()
    document["cases"][0]["kind"] = kind
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(document), encoding="utf-8")
    before = baseline.read_bytes()

    exit_code, report_path = _run(
        tmp_path, baseline=baseline, extra=("--write-baseline",)
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "kind must be one of" in captured.err
    assert "Traceback" not in captured.err
    assert baseline.read_bytes() == before
    assert not report_path.exists()


def test_baseline_kind_change_is_drift_not_regression(tmp_path):
    document = _baseline_document()
    entry = next(
        case for case in document["cases"] if case["id"] == "valid.minimal"
    )
    entry["kind"] = "gateway"
    entry["signature"] = {
        "outcome": "success",
        "status_code": 200,
        "attempts": [],
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(document), encoding="utf-8")

    exit_code, report_path = _run(tmp_path, baseline=baseline)
    report = _read_report(report_path)

    assert exit_code == 0
    assert report["baseline"]["kind_changes"] == ["valid.minimal"]
    assert report["baseline"]["newly_failing"] == []
    assert report["baseline"]["fingerprint_match"] is True
    assert report["result"] == {
        "ok": True,
        "exit_code": 0,
        "regressions": False,
    }


def test_baseline_kind_change_fails_strict_mode(tmp_path):
    document = _baseline_document()
    entry = next(
        case for case in document["cases"] if case["id"] == "valid.minimal"
    )
    entry["kind"] = "gateway"
    entry["signature"] = {
        "outcome": "success",
        "status_code": 200,
        "attempts": [],
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(document), encoding="utf-8")

    exit_code, report_path = _run(
        tmp_path, baseline=baseline, extra=("--strict-baseline",)
    )
    report = _read_report(report_path)

    assert exit_code == 1
    assert report["baseline"]["kind_changes"] == ["valid.minimal"]
    assert report["result"] == {
        "ok": False,
        "exit_code": 1,
        "regressions": False,
    }


# --- Output path collisions ----------------------------------------------


def test_json_out_cannot_equal_baseline(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(REAL_BASELINE.read_bytes())
    before = baseline.read_bytes()

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(baseline),
            "--json-out",
            str(baseline),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be the same file as --baseline" in captured.err
    assert "Traceback" not in captured.err
    assert baseline.read_bytes() == before


@pytest.mark.parametrize(
    "target_name",
    ["report.json", "valid.minimal.json"],
    ids=["new-file", "existing-fixture"],
)
def test_json_out_inside_corpus_is_rejected(tmp_path, capsys, target_name):
    corpus = _copy_corpus(tmp_path)
    before = _snapshot(corpus)
    target = corpus / target_name

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(tmp_path / "missing.json"),
            "--allow-missing-baseline",
            "--json-out",
            str(target),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be inside the corpus directory" in captured.err
    assert "Traceback" not in captured.err
    assert _snapshot(corpus) == before
    assert load_cases(corpus)


def test_json_out_normalized_path_inside_corpus_is_rejected(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    (corpus / "nested").mkdir()
    before = _snapshot(corpus)
    target = corpus / "nested" / ".." / "report.json"

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(tmp_path / "missing.json"),
            "--allow-missing-baseline",
            "--json-out",
            str(target),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be inside the corpus directory" in captured.err
    assert _snapshot(corpus) == before


def test_json_out_symlink_to_corpus_is_rejected(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    link = tmp_path / "corpus-link"
    try:
        link.symlink_to(corpus, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    before = _snapshot(corpus)

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(tmp_path / "missing.json"),
            "--allow-missing-baseline",
            "--json-out",
            str(link / "report.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be inside the corpus directory" in captured.err
    assert _snapshot(corpus) == before


def test_json_out_cannot_be_the_corpus_directory(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(tmp_path / "missing.json"),
            "--allow-missing-baseline",
            "--json-out",
            str(corpus),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be inside the corpus directory" in captured.err


def test_baseline_inside_corpus_is_rejected(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    before = _snapshot(corpus)
    baseline = corpus / "eval_baseline.json"

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(baseline),
            "--allow-missing-baseline",
            "--write-baseline",
            "--json-out",
            str(tmp_path / "report.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be inside the corpus directory" in captured.err
    assert "Traceback" not in captured.err
    assert not baseline.exists()
    assert _snapshot(corpus) == before
    assert load_cases(corpus)


def test_baseline_inside_corpus_fixture_is_rejected(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    fixture = corpus / "valid.minimal.json"
    before = fixture.read_bytes()

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(fixture),
            "--json-out",
            str(tmp_path / "report.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must not be inside the corpus directory" in captured.err
    assert fixture.read_bytes() == before


# --- Safe write ordering --------------------------------------------------


def test_invalid_report_destination_preserves_baseline(tmp_path, capsys):
    corpus = _copy_corpus(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(REAL_BASELINE.read_bytes())
    before = baseline.read_bytes()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(baseline),
            "--write-baseline",
            "--json-out",
            str(blocker / "report.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "parent is not a directory" in captured.err
    assert "Traceback" not in captured.err
    assert baseline.read_bytes() == before


def test_report_write_failure_never_mutates_baseline(
    tmp_path, capsys, monkeypatch
):
    corpus = _copy_corpus(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(REAL_BASELINE.read_bytes())
    before = baseline.read_bytes()
    report = tmp_path / "report.json"

    real_write = corpus_eval.write_json_atomic

    def fail_report(path, document):
        if path == report:
            raise corpus_eval.EvalError(
                f"cannot write {path}: simulated failure"
            )
        return real_write(path, document)

    monkeypatch.setattr(corpus_eval, "write_json_atomic", fail_report)

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--baseline",
            str(baseline),
            "--write-baseline",
            "--json-out",
            str(report),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "simulated failure" in captured.err
    assert "Traceback" not in captured.err
    assert baseline.read_bytes() == before
    assert "wrote baseline" not in captured.err


def test_corrupt_baseline_is_not_replaced_by_write_baseline(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{", encoding="utf-8")
    before = baseline.read_bytes()

    exit_code = main(
        [
            "--baseline",
            str(baseline),
            "--allow-missing-baseline",
            "--write-baseline",
            "--json-out",
            str(tmp_path / "report.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not valid JSON" in captured.err
    assert "Traceback" not in captured.err
    assert baseline.read_bytes() == before
    assert not (tmp_path / "report.json").exists()


# --- Regression attribution -------------------------------------------------


def test_verifier_signature_change_is_attributed(tmp_path):
    corpus = _copy_corpus(tmp_path)
    path = corpus / "valid.minimal.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["body"] = '{"choices":[]}'
    document["expect"] = {"ok": False, "reason": "empty_choices"}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    exit_code, report_path = _run(tmp_path, corpus=corpus)
    report = _read_report(report_path)

    assert exit_code == 0
    assert report["baseline"]["signature_changes"] == ["valid.minimal"]
    detail = report["baseline"]["signature_details"][0]
    assert detail["id"] == "valid.minimal"
    assert detail["expected"] == {"ok": True, "reason": None}
    assert detail["actual"] == {"ok": False, "reason": "empty_choices"}
    assert "ok" in detail["changed_fields"]
    assert "reason" in detail["changed_fields"]
    assert report["baseline"]["category_deltas"]["response_failure_reasons"] == {
        "empty_choices": 1
    }
    assert report["baseline"]["newly_failing"] == []
    assert report["baseline"]["newly_passing"] == []


# --- Malformed baseline input handling -------------------------------------


def test_invalid_utf8_baseline_exits_2(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(b"{\xff}")
    before = baseline.read_bytes()

    exit_code, report_path = _run(tmp_path, baseline=baseline)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Traceback" not in captured.err
    assert "cannot read baseline" in captured.err or "invalid baseline" in captured.err
    assert not report_path.exists()
    assert baseline.read_bytes() == before


def test_over_limit_integer_baseline_exits_2(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"schema_version": '
        + ("9" * 5000)
        + ', "corpus": {"case_count": 0, "fingerprint": "sha256:'
        + ("0" * 64)
        + '"}, "cases": []}',
        encoding="utf-8",
    )
    before = baseline.read_bytes()

    exit_code, report_path = _run(tmp_path, baseline=baseline)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Traceback" not in captured.err
    assert "invalid baseline" in captured.err
    assert not report_path.exists()
    assert baseline.read_bytes() == before
