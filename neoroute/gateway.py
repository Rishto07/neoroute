from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
import httpx
from fastapi import HTTPException, Response
from neoroute.attempts import AttemptRecord, ErrorCategory, Outcome, classify_exception, classify_status
from neoroute.config import ProviderConfig, Settings
from neoroute.routing import build_plan
from neoroute.schemas import ChatCompletionRequest

logger = logging.getLogger("neoroute.gateway")


@dataclass
class UpstreamResult:
    body: bytes
    status_code: int
    content_type: str
    provider_name: str
    attempts: int


def _eligible_for_fallback(status_code: int) -> bool:
    # Ordinary 4xx errors are not retried. 429 and 5xx can be transient.
    return status_code == 429 or status_code >= 500


def _record_attempt(
    records: list[AttemptRecord],
    trace_id: str,
    provider: ProviderConfig,
    attempt: int,
    attempt_started: float,
    attempt_wall: datetime,
    outcome: Outcome,
    error_category: ErrorCategory | None = None,
    status_code: int | None = None,
    fallback_attempted: bool = False,
) -> None:
    """Append one attempt record and log it.

    Instrumentation only: fixed typed fields (never secrets, prompts, or
    completions) and it must not alter the response returned to the caller.
    """
    record = AttemptRecord(
        trace_id=trace_id,
        provider=provider.name,
        model=provider.model,
        attempt=attempt,
        started_at=attempt_wall,
        duration_ms=round((time.perf_counter() - attempt_started) * 1000, 3),
        outcome=outcome,
        status_code=status_code,
        error_category=error_category,
        fallback_attempted=fallback_attempted,
    )
    records.append(record)
    logger.info("provider_attempt %s", record)


async def forward_chat(
    payload: ChatCompletionRequest,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    attempt_log: list[AttemptRecord] | None = None,
):
    if payload.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet; send stream=false in this MVP.")
    owned_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    records: list[AttemptRecord] = attempt_log if attempt_log is not None else []
    errors: list[dict[str, str | int]] = []
    try:
        plan = build_plan(settings.providers, payload.model, settings.max_attempts)
        for attempt, provider in enumerate(plan.providers, start=1):
            attempt_started = time.perf_counter()
            attempt_wall = datetime.now(UTC)
            has_next = attempt < len(plan.providers)
            if not provider.api_key:
                errors.append({"provider": provider.name, "error": "missing_api_key"})
                _record_attempt(
                    records, trace_id, provider, attempt, attempt_started, attempt_wall,
                    Outcome.FAILURE, error_category=ErrorCategory.MISSING_API_KEY,
                    fallback_attempted=has_next,
                )
                continue
            request_data = payload.model_dump(exclude={"stream"})
            request_data["model"] = provider.model
            try:
                response = await http_client.post(
                    f"{provider.base_url}/chat/completions",
                    json=request_data,
                    headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
                )
            except httpx.TransportError as exc:
                category = classify_exception(exc)
                errors.append({"provider": provider.name, "error": type(exc).__name__})
                _record_attempt(
                    records, trace_id, provider, attempt, attempt_started, attempt_wall,
                    Outcome.TIMEOUT if category is ErrorCategory.TIMEOUT else Outcome.FAILURE,
                    error_category=category,
                    fallback_attempted=has_next,
                )
                continue
            if response.is_success:
                elapsed = time.perf_counter() - started
                _record_attempt(
                    records, trace_id, provider, attempt, attempt_started, attempt_wall,
                    Outcome.SUCCESS, status_code=response.status_code,
                )
                return UpstreamResult(response.content, response.status_code, response.headers.get("content-type", "application/json"), provider.name, attempt), elapsed
            errors.append({"provider": provider.name, "error": f"http_{response.status_code}"})
            eligible = _eligible_for_fallback(response.status_code)
            _record_attempt(
                records, trace_id, provider, attempt, attempt_started, attempt_wall,
                Outcome.FAILURE, error_category=classify_status(response.status_code),
                status_code=response.status_code,
                fallback_attempted=eligible and has_next,
            )
            if not eligible:
                raise HTTPException(status_code=response.status_code, detail={"message": "Upstream rejected the request; fallback stopped", "provider": provider.name, "upstream_status": response.status_code})
        raise HTTPException(status_code=502, detail={"message": "All eligible providers failed", "attempts": errors})
    finally:
        if owned_client:
            await http_client.aclose()


def attach_trace_headers(response: Response, result: UpstreamResult, elapsed: float) -> None:
    response.headers["X-neoroute-Provider"] = result.provider_name
    response.headers["X-neoroute-Attempts"] = str(result.attempts)
    response.headers["X-neoroute-Latency-Ms"] = str(round(elapsed * 1000))

