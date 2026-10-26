"""Replay the response-regression corpus through the real verifier and gateway.

Every fixture is synthetic and deterministic. Gateway cases execute in-process
with ``httpx.MockTransport``: no provider is contacted, no credential is read,
and no live network access is possible.
"""
from __future__ import annotations

import re

import httpx
import pytest
from corpus_loader import (
    FIXTURE_DIR,
    AttemptExpectation,
    Case,
    FixtureError,
    load_cases,
    validate_case,
)
from corpus_support import (
    LEAK_MARKER,
    SYNTHETIC_PROVIDER_NAMES,
    UNEXERCISED_CATEGORIES,
    build_handler,
    build_payload,
    build_settings,
)
from fastapi import HTTPException

from neoroute.attempts import AttemptRecord, ErrorCategory, Outcome
from neoroute.gateway import forward_chat
from neoroute.response_contract import ResponseFailureReason, check_response

CASES = load_cases()
VERIFY_CASES = tuple(case for case in CASES if case.kind == "verify")
GATEWAY_CASES = tuple(case for case in CASES if case.kind == "gateway")

# These checks catch well-known credential prefixes, explicit auth headers,
# URLs, and email-shaped text in fixture files. They are a guard against
# accidental copies, not proof that a case is synthetic: novel token formats,
# bare high-entropy strings, base64-encoded secrets, and non-email personal data
# are not detectable here. Fixtures are authored, not captured, and reviewed as
# part of the diff.
CREDENTIAL_PATTERNS = {
    "openai_style_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "email_address": re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
}


# --- Loader and corpus integrity ------------------------------------------


def test_corpus_has_verifier_and_gateway_cases():
    assert CASES
    assert VERIFY_CASES
    assert GATEWAY_CASES


def test_corpus_has_valid_verifier_cases():
    assert any(
        case.verify is not None and case.verify.ok for case in VERIFY_CASES
    )


def test_case_ids_are_unique():
    case_ids = [case.id for case in CASES]
    assert len(case_ids) == len(set(case_ids))


def test_filenames_match_case_ids():
    assert all(case.path.stem == case.id for case in CASES)


def test_all_fixture_files_are_discovered():
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    assert {path.stem for path in paths} == {case.id for case in CASES}


def test_loading_is_deterministic_and_ordered():
    reloaded = load_cases()
    assert [case.id for case in reloaded] == [case.id for case in CASES]
    assert [case.id for case in CASES] == sorted(case.id for case in CASES)


def _valid_verify_fixture() -> dict:
    return {
        "id": "sample",
        "kind": "verify",
        "description": "Sample fixture.",
        "expects_json_object": False,
        "body": "{}",
        "expect": {"ok": False, "reason": "missing_choices"},
    }


def test_loader_rejects_unknown_keys(tmp_path):
    raw = _valid_verify_fixture()
    raw["extra"] = True
    with pytest.raises(FixtureError, match="unknown key"):
        validate_case(raw, tmp_path / "sample.json")


def test_loader_rejects_filename_mismatch(tmp_path):
    with pytest.raises(FixtureError, match="must match filename"):
        validate_case(_valid_verify_fixture(), tmp_path / "other.json")


def test_loader_rejects_unknown_reason(tmp_path):
    raw = _valid_verify_fixture()
    raw["expect"]["reason"] = "not_a_reason"
    with pytest.raises(FixtureError, match="unknown response failure reason"):
        validate_case(raw, tmp_path / "sample.json")


def test_loader_rejects_non_boolean_expectation(tmp_path):
    raw = _valid_verify_fixture()
    raw["expect"]["ok"] = "yes"
    with pytest.raises(FixtureError, match="expect.ok"):
        validate_case(raw, tmp_path / "sample.json")


def test_loader_rejects_json_mode_mismatch(tmp_path):
    raw = {
        "id": "sample",
        "kind": "gateway",
        "description": "Sample fixture.",
        "expects_json_object": True,
        "request": {},
        "providers": [{"name": "alpha"}],
        "responses": [{"status": 200, "body": "{}"}],
        "expect": {
            "outcome": "error",
            "status_code": 502,
            "attempts": [
                {
                    "outcome": "failure",
                    "status_code": 200,
                    "error_category": "malformed_response",
                    "fallback_attempted": False,
                }
            ],
        },
    }
    with pytest.raises(FixtureError, match="expects_json_object must match"):
        validate_case(raw, tmp_path / "sample.json")


def _valid_gateway_fixture() -> dict:
    body = '{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
    return {
        "id": "sample",
        "kind": "gateway",
        "description": "Sample fixture.",
        "expects_json_object": False,
        "request": {},
        "providers": [{"name": "alpha"}, {"name": "beta"}],
        "max_attempts": 2,
        "responses": [
            {
                "status": 500,
                "content_type": "application/json",
                "body": '{"error":"unavailable"}',
            },
            {
                "status": 200,
                "content_type": "application/json",
                "body": body,
            },
        ],
        "expect": {
            "outcome": "success",
            "status_code": 200,
            "attempts": [
                {
                    "outcome": "failure",
                    "status_code": 500,
                    "error_category": "provider_server_error",
                    "fallback_attempted": True,
                },
                {
                    "outcome": "success",
                    "status_code": 200,
                    "error_category": None,
                    "fallback_attempted": False,
                },
            ],
            "provider_index": 1,
            "content_type": "application/json",
            "body_encoding": "utf-8",
            "body": body,
        },
    }


def test_loader_rejects_success_body_mismatch(tmp_path):
    raw = _valid_gateway_fixture()
    raw["expect"]["body"] = '{"choices":[]}'
    with pytest.raises(FixtureError, match="success expectations must match"):
        validate_case(raw, tmp_path / "sample.json")


def test_loader_rejects_provider_index_mismatch(tmp_path):
    raw = _valid_gateway_fixture()
    raw["expect"]["provider_index"] = 0
    with pytest.raises(FixtureError, match="must reference the final attempt"):
        validate_case(raw, tmp_path / "sample.json")


def test_loader_rejects_attempts_beyond_routing_plan(tmp_path):
    raw = _valid_gateway_fixture()
    raw["max_attempts"] = 1
    with pytest.raises(FixtureError, match="cannot exceed the routing plan"):
        validate_case(raw, tmp_path / "sample.json")


# --- Verifier replay ------------------------------------------------------


@pytest.mark.parametrize("case", VERIFY_CASES, ids=lambda case: case.id)
def test_verifier_fixture(case: Case):
    result = check_response(
        case.body, expects_json_object=case.expects_json_object
    )
    assert result.ok is case.verify.ok
    assert result.reason is case.verify.reason


@pytest.mark.parametrize("case", VERIFY_CASES, ids=lambda case: case.id)
def test_verifier_replay_is_deterministic(case: Case):
    first = check_response(
        case.body, expects_json_object=case.expects_json_object
    )
    second = check_response(
        case.body, expects_json_object=case.expects_json_object
    )
    assert first == second


# --- Gateway replay -------------------------------------------------------


@pytest.mark.parametrize("case", GATEWAY_CASES, ids=lambda case: case.id)
async def test_gateway_fixture(case: Case, monkeypatch):
    calls: list[str] = []
    records: list[AttemptRecord] = []
    transport = httpx.MockTransport(build_handler(case, calls))

    async with httpx.AsyncClient(transport=transport) as client:
        if case.gateway.outcome == "success":
            result, _ = await forward_chat(
                build_payload(case),
                build_settings(case, monkeypatch),
                client=client,
                attempt_log=records,
            )
        else:
            with pytest.raises(HTTPException) as exc_info:
                await forward_chat(
                    build_payload(case),
                    build_settings(case, monkeypatch),
                    client=client,
                    attempt_log=records,
                )

    if case.gateway.outcome == "success":
        expected_provider = case.providers[case.gateway.provider_index]
        assert result.provider_name == expected_provider.name
        assert result.status_code == case.gateway.status_code
        assert result.content_type == case.gateway.content_type
        assert result.body == case.gateway.body
        assert result.attempts == len(case.gateway.attempts)
    else:
        assert exc_info.value.status_code == case.gateway.status_code
        assert LEAK_MARKER not in str(exc_info.value.detail)

    assert len(records) == len(case.gateway.attempts)
    for record, expected in zip(records, case.gateway.attempts, strict=True):
        assert record.provider == case.providers[record.attempt - 1].name
        assert record.outcome is expected.outcome
        assert record.status_code == expected.status_code
        assert record.error_category is expected.error_category
        assert record.fallback_attempted is expected.fallback_attempted

    expected_calls = sum(
        1
        for expected in case.gateway.attempts
        if expected.error_category is not ErrorCategory.MISSING_API_KEY
    )
    assert len(calls) == expected_calls


# --- Coverage and sanitization -------------------------------------------


def test_response_failure_reasons_are_complete():
    covered = {
        case.verify.reason
        for case in VERIFY_CASES
        if case.verify is not None and case.verify.reason is not None
    }
    assert covered == set(ResponseFailureReason)


def test_error_categories_are_complete_or_documented():
    covered = {
        attempt.error_category
        for case in GATEWAY_CASES
        if case.gateway is not None
        for attempt in case.gateway.attempts
        if attempt.error_category is not None
    }
    assert covered & set(UNEXERCISED_CATEGORIES) == set()
    assert covered | set(UNEXERCISED_CATEGORIES) == set(ErrorCategory)


# Three distinct concepts are measured separately below:
#   1. per-attempt Outcome (neoroute.attempts.Outcome): success | failure | timeout
#   2. terminal gateway result: UpstreamResult (fixture "success") or HTTPException
#      (fixture "error"), proven by test_gateway_fixture
#   3. fallback success: a terminal success reached after a non-success attempt


def _attempt_expectations() -> list[tuple[Case, AttemptExpectation]]:
    return [
        (case, attempt)
        for case in GATEWAY_CASES
        if case.gateway is not None
        for attempt in case.gateway.attempts
    ]


def test_per_attempt_outcomes_are_complete():
    evidence: dict[Outcome, set[str]] = {}
    for case, attempt in _attempt_expectations():
        evidence.setdefault(attempt.outcome, set()).add(case.id)

    assert set(evidence) == set(Outcome)
    for outcome in Outcome:
        assert evidence.get(
            outcome
        ), f"no fixture exercises per-attempt outcome {outcome.value}"


def test_terminal_gateway_results_are_complete():
    terminal = [
        (case.id, case.gateway.outcome)
        for case in GATEWAY_CASES
        if case.gateway is not None
    ]
    success_ids = {case_id for case_id, result in terminal if result == "success"}
    error_ids = {case_id for case_id, result in terminal if result == "error"}

    assert success_ids
    assert error_ids
    assert success_ids | error_ids == {case_id for case_id, _ in terminal}


def test_fallback_success_is_distinct_from_immediate_success():
    immediate: list[str] = []
    fallback: list[str] = []

    for case in GATEWAY_CASES:
        if case.gateway is None or case.gateway.outcome != "success":
            continue
        attempts = case.gateway.attempts
        assert attempts[-1].outcome is Outcome.SUCCESS
        if len(attempts) == 1:
            immediate.append(case.id)
        else:
            assert all(
                attempt.outcome is not Outcome.SUCCESS
                for attempt in attempts[:-1]
            )
            fallback.append(case.id)

    assert immediate
    assert fallback
    assert set(immediate).isdisjoint(fallback)


def test_fallback_flags_cover_continue_and_stop():
    flags = {
        attempt.fallback_attempted for _, attempt in _attempt_expectations()
    }
    assert flags == {True, False}


def test_fixtures_use_only_synthetic_provider_names():
    for case in CASES:
        for spec in case.providers:
            assert spec.name in SYNTHETIC_PROVIDER_NAMES


def test_fixtures_contain_no_known_credentials_auth_headers_or_urls():
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    assert len(paths) == len(CASES)

    findings: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for name, pattern in CREDENTIAL_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.name}: {name}")
        if "authorization" in lowered:
            findings.append(f"{path.name}: authorization header")
        if "bearer " in lowered:
            findings.append(f"{path.name}: bearer credential")
        if "http://" in lowered or "https://" in lowered:
            findings.append(f"{path.name}: url")

    assert findings == []
