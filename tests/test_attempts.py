import dataclasses
import os

import httpx
import pytest
from fastapi import HTTPException, Response

from neoroute.attempts import (
    AttemptRecord,
    ErrorCategory,
    Outcome,
    classify_exception,
    classify_status,
)
from neoroute.config import ProviderConfig, Settings
from neoroute.gateway import UpstreamResult, attach_trace_headers, forward_chat
from neoroute.schemas import ChatCompletionRequest

SUCCESS_BODY = {"id": "chatcmpl-test", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def make_settings():
    os.environ["TEST_KEY_A"] = "key-a"
    os.environ["TEST_KEY_B"] = "key-b"
    return Settings((ProviderConfig("a", "https://a.example/v1", "TEST_KEY_A", "model-a"), ProviderConfig("b", "https://b.example/v1", "TEST_KEY_B", "model-b")), 2, 2)


def make_payload(content: str = "hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": content}])


def test_classify_status_maps_http_categories():
    assert classify_status(429) is ErrorCategory.RATE_LIMITED
    assert classify_status(500) is ErrorCategory.PROVIDER_SERVER_ERROR
    assert classify_status(503) is ErrorCategory.PROVIDER_SERVER_ERROR
    assert classify_status(400) is ErrorCategory.PROVIDER_CLIENT_ERROR
    assert classify_status(401) is ErrorCategory.PROVIDER_CLIENT_ERROR
    assert classify_status(404) is ErrorCategory.PROVIDER_CLIENT_ERROR
    assert classify_status(302) is ErrorCategory.INVALID_RESPONSE


def test_classify_exception_maps_transport_categories():
    assert classify_exception(httpx.ReadTimeout("timed out")) is ErrorCategory.TIMEOUT
    assert classify_exception(httpx.ConnectTimeout("timed out")) is ErrorCategory.TIMEOUT
    assert classify_exception(httpx.ConnectError("refused")) is ErrorCategory.CONNECTION_ERROR
    assert classify_exception(httpx.ProtocolError("broken")) is ErrorCategory.CONNECTION_ERROR
    assert classify_exception(ValueError("boom")) is ErrorCategory.UNKNOWN


@pytest.mark.asyncio
async def test_successful_attempt_is_recorded():
    async def handler(_: httpx.Request):
        return httpx.Response(200, json=SUCCESS_BODY)
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, elapsed = await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert result.provider_name == "a"
    assert elapsed >= 0
    assert len(records) == 1
    record = records[0]
    assert record.trace_id
    assert record.provider == "a"
    assert record.model == "model-a"
    assert record.attempt == 1
    assert record.outcome is Outcome.SUCCESS
    assert record.status_code == 200
    assert record.error_category is None
    assert record.fallback_attempted is False
    assert record.started_at.tzinfo is not None
    assert record.duration_ms >= 0


@pytest.mark.asyncio
async def test_failed_attempt_then_successful_fallback_records_in_order():
    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=SUCCESS_BODY)
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert result.provider_name == "b"
    assert result.attempts == 2
    assert len(records) == 2
    first, second = records
    assert (first.provider, first.attempt) == ("a", 1)
    assert first.outcome is Outcome.FAILURE
    assert first.status_code == 503
    assert first.error_category is ErrorCategory.PROVIDER_SERVER_ERROR
    assert first.fallback_attempted is True
    assert (second.provider, second.attempt) == ("b", 2)
    assert second.outcome is Outcome.SUCCESS
    assert second.status_code == 200
    assert second.fallback_attempted is False
    assert first.trace_id == second.trace_id
    assert first.started_at <= second.started_at


@pytest.mark.asyncio
async def test_rate_limited_attempt_is_recorded_and_falls_back():
    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=SUCCESS_BODY)
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert result.provider_name == "b"
    assert records[0].status_code == 429
    assert records[0].error_category is ErrorCategory.RATE_LIMITED
    assert records[0].fallback_attempted is True
    assert records[1].outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_timeout_attempt_is_recorded_and_falls_back():
    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=SUCCESS_BODY)
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert result.provider_name == "b"
    record = records[0]
    assert record.outcome is Outcome.TIMEOUT
    assert record.error_category is ErrorCategory.TIMEOUT
    assert record.status_code is None
    assert record.fallback_attempted is True
    assert record.duration_ms >= 0


@pytest.mark.parametrize("exc_type", [httpx.RemoteProtocolError, httpx.ProxyError])
@pytest.mark.asyncio
async def test_transport_errors_are_recorded_and_fall_back(exc_type):
    calls: list[str] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "a.example" in str(request.url):
            raise exc_type("transport failure", request=request)
        return httpx.Response(200, json=SUCCESS_BODY)
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert result.provider_name == "b"
    assert result.attempts == 2
    assert len(calls) == 2
    assert len(records) == 2
    first, second = records
    assert first.provider == "a"
    assert first.outcome is Outcome.FAILURE
    assert first.error_category is ErrorCategory.CONNECTION_ERROR
    assert first.status_code is None
    assert first.fallback_attempted is True
    assert second.provider == "b"
    assert second.outcome is Outcome.SUCCESS
    assert second.status_code == 200
    assert second.fallback_attempted is False
    assert first.trace_id == second.trace_id


@pytest.mark.asyncio
async def test_connection_failures_record_every_attempt_in_order():
    calls: list[str] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        raise httpx.ConnectError("refused", request=request)
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HTTPException) as exc:
            await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert exc.value.status_code == 502
    assert len(calls) == 2
    assert [r.attempt for r in records] == [1, 2]
    assert [r.provider for r in records] == ["a", "b"]
    assert all(r.error_category is ErrorCategory.CONNECTION_ERROR for r in records)
    assert all(r.outcome is Outcome.FAILURE for r in records)
    assert records[0].fallback_attempted is True
    assert records[1].fallback_attempted is False


@pytest.mark.asyncio
async def test_non_retryable_400_is_recorded_and_stops_fallback():
    calls: list[str] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(400, json={"error": "bad request"})
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HTTPException) as exc:
            await forward_chat(make_payload(), make_settings(), client=client, attempt_log=records)
    assert exc.value.status_code == 400
    assert len(calls) == 1
    assert len(records) == 1
    record = records[0]
    assert record.error_category is ErrorCategory.PROVIDER_CLIENT_ERROR
    assert record.outcome is Outcome.FAILURE
    assert record.status_code == 400
    assert record.fallback_attempted is False


@pytest.mark.asyncio
async def test_attempt_records_contain_no_secrets_prompts_or_completions():
    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "completion-content"}}]})
    payload = make_payload("prompt-content")
    records: list[AttemptRecord] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await forward_chat(payload, make_settings(), client=client, attempt_log=records)
    blob = repr([dataclasses.asdict(record) for record in records])
    for sensitive in ("key-a", "key-b", "Bearer", "Authorization", "authorization", "prompt-content", "completion-content"):
        assert sensitive not in blob
    assert len(records) == 2


def test_trace_headers_unchanged():
    result = UpstreamResult(body=b"{}", status_code=200, content_type="application/json", provider_name="b", attempts=2)
    response = Response()
    attach_trace_headers(response, result, 0.5)
    assert response.headers["X-neoroute-Provider"] == "b"
    assert response.headers["X-neoroute-Attempts"] == "2"
    assert response.headers["X-neoroute-Latency-Ms"] == "500"
