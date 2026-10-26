"""Strict, dependency-free loader for the replayable response corpus.

Each fixture is a small synthetic JSON document under
``tests/fixtures/response_corpus``. The file stem is the case id, and every
case declares whether it is replayed through the response verifier or through
the gateway. Parsing is intentionally strict: unknown keys, wrong types, and
internally inconsistent expectations fail with the offending fixture path so a
malformed case can never silently weaken coverage.
"""
from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neoroute.attempts import ErrorCategory, Outcome
from neoroute.capabilities import ModelCapabilities
from neoroute.response_contract import ResponseFailureReason

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "response_corpus"

KINDS = frozenset({"verify", "gateway"})
SUPPORTED_ENCODINGS = frozenset({"utf-8", "base64"})
RAISE_KINDS = frozenset({"timeout", "connect_error"})
GATEWAY_OUTCOMES = frozenset({"success", "error"})

VERIFY_KEYS = frozenset(
    {
        "id",
        "kind",
        "description",
        "expects_json_object",
        "body_encoding",
        "body",
        "expect",
    }
)
GATEWAY_KEYS = frozenset(
    {
        "id",
        "kind",
        "description",
        "expects_json_object",
        "request",
        "providers",
        "max_attempts",
        "responses",
        "expect",
    }
)


class FixtureError(ValueError):
    """A corpus fixture is malformed or internally inconsistent."""


@dataclass(frozen=True)
class VerifyExpectation:
    ok: bool
    reason: ResponseFailureReason | None


@dataclass(frozen=True)
class AttemptExpectation:
    outcome: Outcome
    status_code: int | None
    error_category: ErrorCategory | None
    fallback_attempted: bool


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    missing_api_key: bool
    capabilities: Mapping[str, str]


@dataclass(frozen=True)
class GatewayResponse:
    status_code: int | None
    content_type: str | None
    body: bytes | None
    raise_kind: str | None


@dataclass(frozen=True)
class GatewayExpectation:
    outcome: str
    status_code: int
    provider_index: int | None
    content_type: str | None
    body: bytes | None
    attempts: tuple[AttemptExpectation, ...]


@dataclass(frozen=True)
class Case:
    id: str
    path: Path
    kind: str
    description: str
    expects_json_object: bool
    body: bytes | None = None
    verify: VerifyExpectation | None = None
    request: Mapping[str, Any] = field(default_factory=dict)
    providers: tuple[ProviderSpec, ...] = ()
    max_attempts: int = 0
    responses: tuple[GatewayResponse, ...] = ()
    gateway: GatewayExpectation | None = None


def _fail(path: Path, message: str) -> FixtureError:
    return FixtureError(f"{path}: {message}")


def _require_mapping(value: Any, path: Path, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, f"{label} must be a JSON object")
    return value


def _require_text(
    mapping: Mapping[str, Any], key: str, path: Path, label: str
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, f"{label} must be a non-empty string")
    return value


def _require_bool(
    mapping: Mapping[str, Any], key: str, path: Path, label: str
) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise _fail(path, f"{label} must be a boolean")
    return value


def _require_int(
    value: Any,
    path: Path,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(path, f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise _fail(path, f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise _fail(path, f"{label} must be <= {maximum}")
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: frozenset[str],
    path: Path,
    label: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise _fail(path, f"{label} has unknown key(s): {', '.join(unknown)}")


def _decode_body(
    mapping: Mapping[str, Any], path: Path, label: str
) -> bytes:
    encoding = mapping.get("body_encoding", "utf-8")
    if encoding not in SUPPORTED_ENCODINGS:
        raise _fail(
            path,
            f"{label}.body_encoding must be one of {sorted(SUPPORTED_ENCODINGS)}",
        )
    body = mapping.get("body")
    if not isinstance(body, str):
        raise _fail(path, f"{label}.body must be a string")
    if encoding == "utf-8":
        try:
            return body.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _fail(path, f"{label}.body is not valid UTF-8 text") from exc
    try:
        return base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _fail(path, f"{label}.body is not valid base64") from exc


def _parse_reason(value: Any, path: Path) -> ResponseFailureReason:
    if not isinstance(value, str):
        raise _fail(path, "expect.reason must be a string or null")
    try:
        return ResponseFailureReason(value)
    except ValueError as exc:
        raise _fail(path, f"unknown response failure reason: {value!r}") from exc


def _parse_category(value: Any, path: Path) -> ErrorCategory:
    if not isinstance(value, str):
        raise _fail(path, "error_category must be a string")
    try:
        return ErrorCategory(value)
    except ValueError as exc:
        raise _fail(path, f"unknown error category: {value!r}") from exc


def _parse_verify(raw: Mapping[str, Any], path: Path) -> VerifyExpectation:
    expect = _require_mapping(raw.get("expect"), path, "expect")
    _reject_unknown_keys(expect, frozenset({"ok", "reason"}), path, "expect")
    ok = _require_bool(expect, "ok", path, "expect.ok")
    if ok:
        if expect.get("reason") is not None:
            raise _fail(path, "expect.reason must be null when expect.ok is true")
        return VerifyExpectation(ok=True, reason=None)
    if expect.get("reason") is None:
        raise _fail(path, "expect.reason is required when expect.ok is false")
    return VerifyExpectation(ok=False, reason=_parse_reason(expect["reason"], path))


def _parse_attempt(
    raw: Any, path: Path, index: int, *, last: bool
) -> AttemptExpectation:
    label = f"expect.attempts[{index}]"
    item = _require_mapping(raw, path, label)
    _reject_unknown_keys(
        item,
        frozenset({"outcome", "status_code", "error_category", "fallback_attempted"}),
        path,
        label,
    )

    outcome_value = item.get("outcome")
    if not isinstance(outcome_value, str):
        raise _fail(path, f"{label}.outcome must be a string")
    try:
        outcome = Outcome(outcome_value)
    except ValueError as exc:
        raise _fail(path, f"{label}: unknown outcome {outcome_value!r}") from exc

    status_code = item.get("status_code")
    if status_code is not None:
        status_code = _require_int(
            status_code, path, f"{label}.status_code", minimum=100, maximum=599
        )

    category_value = item.get("error_category")
    if outcome is Outcome.SUCCESS:
        if category_value is not None:
            raise _fail(path, f"{label}.error_category must be null for a success")
        if status_code is None or not 200 <= status_code < 300:
            raise _fail(path, f"{label}.status_code must be a 2xx for a success")
        error_category = None
    else:
        if category_value is None:
            raise _fail(
                path, f"{label}.error_category is required for {outcome.value!r}"
            )
        error_category = _parse_category(category_value, path)

    fallback_attempted = _require_bool(
        item, "fallback_attempted", path, f"{label}.fallback_attempted"
    )
    if last and fallback_attempted:
        raise _fail(
            path, f"{label}.fallback_attempted must be false for the final attempt"
        )

    return AttemptExpectation(
        outcome=outcome,
        status_code=status_code,
        error_category=error_category,
        fallback_attempted=fallback_attempted,
    )


def _parse_providers(
    raw: Mapping[str, Any], path: Path
) -> tuple[ProviderSpec, ...]:
    providers_raw = raw.get("providers")
    if not isinstance(providers_raw, list) or not providers_raw:
        raise _fail(path, "providers must be a non-empty array")

    providers: list[ProviderSpec] = []
    names: set[str] = set()
    for index, entry in enumerate(providers_raw):
        label = f"providers[{index}]"
        item = _require_mapping(entry, path, label)
        _reject_unknown_keys(
            item,
            frozenset({"name", "missing_api_key", "capabilities"}),
            path,
            label,
        )
        name = _require_text(item, "name", path, f"{label}.name")
        if name in names:
            raise _fail(path, f"duplicate provider name: {name!r}")
        names.add(name)

        missing_api_key = item.get("missing_api_key", False)
        if not isinstance(missing_api_key, bool):
            raise _fail(path, f"{label}.missing_api_key must be a boolean")

        capabilities = _require_mapping(
            item.get("capabilities", {}), path, f"{label}.capabilities"
        )
        try:
            ModelCapabilities.from_mapping(capabilities)
        except (TypeError, ValueError) as exc:
            raise _fail(path, f"{label}.capabilities is invalid: {exc}") from exc

        providers.append(
            ProviderSpec(
                name=name,
                missing_api_key=missing_api_key,
                capabilities=dict(capabilities),
            )
        )
    return tuple(providers)


def _parse_responses(
    raw: Mapping[str, Any], path: Path, provider_count: int
) -> tuple[GatewayResponse, ...]:
    responses_raw = raw.get("responses")
    if not isinstance(responses_raw, list):
        raise _fail(path, "responses must be an array")
    if len(responses_raw) != provider_count:
        raise _fail(path, "responses must contain exactly one entry per provider")

    responses: list[GatewayResponse] = []
    for index, entry in enumerate(responses_raw):
        label = f"responses[{index}]"
        item = _require_mapping(entry, path, label)

        if "raise" in item:
            _reject_unknown_keys(item, frozenset({"raise"}), path, label)
            raise_kind = item.get("raise")
            if raise_kind not in RAISE_KINDS:
                raise _fail(
                    path, f"{label}.raise must be one of {sorted(RAISE_KINDS)}"
                )
            responses.append(
                GatewayResponse(
                    status_code=None,
                    content_type=None,
                    body=None,
                    raise_kind=raise_kind,
                )
            )
            continue

        _reject_unknown_keys(
            item,
            frozenset({"status", "content_type", "body_encoding", "body"}),
            path,
            label,
        )
        status_code = _require_int(
            item.get("status"), path, f"{label}.status", minimum=100, maximum=599
        )
        content_type = item.get("content_type", "application/json")
        if not isinstance(content_type, str) or not content_type.strip():
            raise _fail(path, f"{label}.content_type must be a non-empty string")
        body = _decode_body(item, path, label)
        responses.append(
            GatewayResponse(
                status_code=status_code,
                content_type=content_type,
                body=body,
                raise_kind=None,
            )
        )
    return tuple(responses)


def _parse_request(
    raw: Mapping[str, Any], path: Path, expects_json_object: bool
) -> dict[str, Any]:
    request = _require_mapping(raw.get("request", {}), path, "request")
    _reject_unknown_keys(
        request, frozenset({"response_format", "tools"}), path, "request"
    )
    if "response_format" in request:
        _require_mapping(
            request["response_format"], path, "request.response_format"
        )
    tools = request.get("tools", False)
    if not isinstance(tools, bool):
        raise _fail(path, "request.tools must be a boolean")

    response_format = request.get("response_format")
    json_object = (
        isinstance(response_format, Mapping)
        and response_format.get("type") == "json_object"
    )
    if json_object is not expects_json_object:
        raise _fail(
            path, "expects_json_object must match request.response_format"
        )
    return dict(request)


def _parse_gateway_expect(
    raw: Mapping[str, Any],
    path: Path,
    provider_count: int,
) -> GatewayExpectation:
    expect = _require_mapping(raw.get("expect"), path, "expect")
    outcome = expect.get("outcome")
    if outcome not in GATEWAY_OUTCOMES:
        raise _fail(
            path, f"expect.outcome must be one of {sorted(GATEWAY_OUTCOMES)}"
        )

    attempts_raw = expect.get("attempts")
    if not isinstance(attempts_raw, list) or not attempts_raw:
        raise _fail(path, "expect.attempts must be a non-empty array")
    attempts = tuple(
        _parse_attempt(item, path, index, last=index == len(attempts_raw) - 1)
        for index, item in enumerate(attempts_raw)
    )

    status_code = _require_int(
        expect.get("status_code"),
        path,
        "expect.status_code",
        minimum=100,
        maximum=599,
    )

    if outcome == "success":
        _reject_unknown_keys(
            expect,
            frozenset(
                {
                    "outcome",
                    "status_code",
                    "provider_index",
                    "content_type",
                    "body_encoding",
                    "body",
                    "attempts",
                }
            ),
            path,
            "expect",
        )
        provider_index = _require_int(
            expect.get("provider_index"),
            path,
            "expect.provider_index",
            minimum=0,
            maximum=provider_count - 1,
        )
        content_type = expect.get("content_type")
        if not isinstance(content_type, str) or not content_type.strip():
            raise _fail(path, "expect.content_type must be a non-empty string")
        body = _decode_body(expect, path, "expect")
        if attempts[-1].outcome is not Outcome.SUCCESS:
            raise _fail(
                path, "expect.attempts must end with a success for a success outcome"
            )
        return GatewayExpectation(
            outcome="success",
            status_code=status_code,
            provider_index=provider_index,
            content_type=content_type,
            body=body,
            attempts=attempts,
        )

    _reject_unknown_keys(
        expect, frozenset({"outcome", "status_code", "attempts"}), path, "expect"
    )
    if attempts[-1].outcome is Outcome.SUCCESS:
        raise _fail(
            path, "expect.attempts must not end with a success for an error outcome"
        )
    return GatewayExpectation(
        outcome="error",
        status_code=status_code,
        provider_index=None,
        content_type=None,
        body=None,
        attempts=attempts,
    )


def validate_case(raw: Any, path: Path) -> Case:
    """Validate one fixture document and return its typed representation."""
    if not isinstance(raw, Mapping):
        raise _fail(path, "fixture must be a JSON object")

    kind = raw.get("kind")
    if kind not in KINDS:
        raise _fail(path, f"kind must be one of {sorted(KINDS)}")

    case_id = _require_text(raw, "id", path, "id")
    if case_id != path.stem:
        raise _fail(
            path, f"id {case_id!r} must match filename {path.stem!r}"
        )
    description = _require_text(raw, "description", path, "description")
    expects_json_object = _require_bool(
        raw, "expects_json_object", path, "expects_json_object"
    )

    if kind == "verify":
        _reject_unknown_keys(raw, VERIFY_KEYS, path, "fixture")
        if "body" not in raw:
            raise _fail(path, "body is required for verify fixtures")
        return Case(
            id=case_id,
            path=path,
            kind=kind,
            description=description,
            expects_json_object=expects_json_object,
            body=_decode_body(raw, path, "fixture"),
            verify=_parse_verify(raw, path),
        )

    _reject_unknown_keys(raw, GATEWAY_KEYS, path, "fixture")
    request = _parse_request(raw, path, expects_json_object)
    providers = _parse_providers(raw, path)
    max_attempts = _require_int(
        raw.get("max_attempts", len(providers)),
        path,
        "max_attempts",
        minimum=1,
        maximum=len(providers),
    )
    responses = _parse_responses(raw, path, len(providers))
    gateway = _parse_gateway_expect(raw, path, len(providers))

    if len(gateway.attempts) > min(max_attempts, len(providers)):
        raise _fail(path, "expect.attempts cannot exceed the routing plan")

    if gateway.outcome == "success":
        # Success is terminal and occurs on the final attempted provider. The
        # corpus does not use capability filtering, which would shift indices.
        if gateway.provider_index != len(gateway.attempts) - 1:
            raise _fail(
                path, "expect.provider_index must reference the final attempt"
            )
        expected = responses[gateway.provider_index]
        if expected.raise_kind is not None:
            raise _fail(
                path, "a successful provider_index must not raise a transport error"
            )
        if (
            gateway.status_code,
            gateway.content_type,
            gateway.body,
        ) != (
            expected.status_code,
            expected.content_type,
            expected.body,
        ):
            raise _fail(
                path, "success expectations must match the provider response"
            )

    return Case(
        id=case_id,
        path=path,
        kind=kind,
        description=description,
        expects_json_object=expects_json_object,
        request=request,
        providers=providers,
        max_attempts=max_attempts,
        responses=responses,
        gateway=gateway,
    )


def load_cases(directory: Path = FIXTURE_DIR) -> list[Case]:
    """Load and validate every fixture, ordered by filename."""
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FixtureError(f"No corpus fixtures found in {directory}")

    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise FixtureError(f"{path}: cannot read fixture: {exc}") from exc
        case = validate_case(raw, path)
        previous = seen.get(case.id)
        if previous is not None:
            raise FixtureError(
                f"{path}: duplicate case id {case.id!r} also defined by {previous}"
            )
        seen[case.id] = path
        cases.append(case)
    return cases
