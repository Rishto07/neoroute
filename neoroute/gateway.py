from __future__ import annotations
import time
from dataclasses import dataclass
import httpx
from fastapi import HTTPException, Response
from neoroute.config import Settings
from neoroute.routing import build_plan
from neoroute.schemas import ChatCompletionRequest


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


async def forward_chat(payload: ChatCompletionRequest, settings: Settings, client: httpx.AsyncClient | None = None):
    if payload.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet; send stream=false in this MVP.")
    owned_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)
    started = time.perf_counter()
    errors: list[dict[str, str | int]] = []
    try:
        plan = build_plan(settings.providers, payload.model, settings.max_attempts)
        for attempt, provider in enumerate(plan.providers, start=1):
            if not provider.api_key:
                errors.append({"provider": provider.name, "error": "missing_api_key"})
                continue
            request_data = payload.model_dump(exclude={"stream"})
            request_data["model"] = provider.model
            try:
                response = await http_client.post(
                    f"{provider.base_url}/chat/completions",
                    json=request_data,
                    headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                errors.append({"provider": provider.name, "error": type(exc).__name__})
                continue
            if response.is_success:
                elapsed = time.perf_counter() - started
                return UpstreamResult(response.content, response.status_code, response.headers.get("content-type", "application/json"), provider.name, attempt), elapsed
            errors.append({"provider": provider.name, "error": f"http_{response.status_code}"})
            if not _eligible_for_fallback(response.status_code):
                raise HTTPException(status_code=response.status_code, detail={"message": "Upstream rejected the request; fallback stopped", "provider": provider.name, "upstream_status": response.status_code})
        raise HTTPException(status_code=502, detail={"message": "All eligible providers failed", "attempts": errors})
    finally:
        if owned_client:
            await http_client.aclose()


def attach_trace_headers(response: Response, result: UpstreamResult, elapsed: float) -> None:
    response.headers["X-neoroute-Provider"] = result.provider_name
    response.headers["X-neoroute-Attempts"] = str(result.attempts)
    response.headers["X-neoroute-Latency-Ms"] = str(round(elapsed * 1000))

