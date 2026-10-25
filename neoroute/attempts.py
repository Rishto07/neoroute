"""Stable error taxonomy and provider-attempt records for Neoroute.

Records are internal instrumentation: they carry fixed, typed fields only and
are never part of a public response format. Credentials, authorization
headers, prompts, and completion contents must never be stored on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import httpx


class ErrorCategory(StrEnum):
    """Normalized classification for a provider attempt outcome.

    Categories are a stable contract: values must not change across
    milestones. Fallback eligibility stays governed by the gateway's
    existing status rules (429 and 5xx retry, other 4xx stop).
    """

    RATE_LIMITED = "rate_limited"                    # HTTP 429
    PROVIDER_SERVER_ERROR = "provider_server_error"  # HTTP 5xx
    PROVIDER_CLIENT_ERROR = "provider_client_error"  # non-retryable HTTP 4xx (429 handled above)
    TIMEOUT = "timeout"                              # provider request timed out
    CONNECTION_ERROR = "connection_error"            # connection or transport failure
    MISSING_API_KEY = "missing_api_key"              # local configuration problem; provider skipped
    INVALID_RESPONSE = "invalid_response"            # non-2xx that yields no usable completion (e.g. 3xx)
    UNKNOWN = "unknown"                              # unclassified failure


class Outcome(StrEnum):
    """Result of a single provider attempt."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


def classify_status(status_code: int) -> ErrorCategory:
    """Map an upstream HTTP status code to a stable error category."""
    if status_code == 429:
        return ErrorCategory.RATE_LIMITED
    if status_code >= 500:
        return ErrorCategory.PROVIDER_SERVER_ERROR
    if 400 <= status_code < 500:
        return ErrorCategory.PROVIDER_CLIENT_ERROR
    # 1xx/3xx and other non-2xx statuses: the upstream produced no usable
    # completion. Safely detectable from the status alone.
    return ErrorCategory.INVALID_RESPONSE


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Map a transport exception raised by the HTTP client to a stable category."""
    if isinstance(exc, httpx.TimeoutException):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, httpx.TransportError):
        return ErrorCategory.CONNECTION_ERROR
    return ErrorCategory.UNKNOWN


@dataclass(frozen=True)
class AttemptRecord:
    """One provider attempt within a single gateway request.

    ``started_at`` is a timezone-aware (UTC) wall-clock timestamp;
    ``duration_ms`` is measured with a monotonic clock. ``fallback_attempted``
    means routing continued to another provider after this attempt.
    No credentials, authorization headers, prompts, or completion contents.
    """

    trace_id: str
    provider: str
    model: str
    attempt: int
    started_at: datetime
    duration_ms: float
    outcome: Outcome
    status_code: int | None = None
    error_category: ErrorCategory | None = None
    fallback_attempted: bool = False
