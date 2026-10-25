import json
import logging
import os
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from neoroute.attempts import AttemptRecord, ErrorCategory, Outcome
from neoroute.capabilities import ModelCapabilities
from neoroute.config import ProviderConfig, Settings
from neoroute.gateway import forward_chat
from neoroute.schemas import ChatCompletionRequest

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"id": 1}'},
    }
]


def caps(**states):
    return ModelCapabilities.from_mapping(states)


def provider(name, *, capabilities=None):
    env_name = f"TEST_M3_KEY_{name.upper()}"
    os.environ[env_name] = f"key-{name}"
    return ProviderConfig(
        name=name,
        base_url=f"https://{name}.example/v1",
        api_key_env=env_name,
        model=f"model-{name}",
        capabilities=(
            capabilities if capabilities is not None else ModelCapabilities()
        ),
    )


def make_settings(*providers, max_attempts=2):
    return Settings(
        providers=tuple(providers),
        timeout_seconds=2,
        max_attempts=max_attempts,
    )


def make_payload(**extra):
    return ChatCompletionRequest(
        model="auto",
        messages=[{"role": "user", "content": "hello"}],
        **extra,
    )


def image_payload():
    return ChatCompletionRequest(
        model="auto",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/x.png"},
                    }
                ],
            }
        ],
    )


def completion(content="ok", **message_fields):
    message = {"role": "assistant", "content": content}
    message.update(message_fields)
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def json_response(document, *, status_code=200, content_type="application/json"):
    return httpx.Response(
        status_code,
        content=json.dumps(document).encode("utf-8"),
        headers={"content-type": content_type},
    )


@pytest.mark.asyncio
async def test_malformed_success_falls_back_and_records_attempt():
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "a.example" in str(request.url):
            return httpx.Response(
                200,
                content=b'{"unexpected": true}',
                headers={"content-type": "application/json"},
            )
        return json_response(completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(),
            make_settings(provider("a"), provider("b")),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.status_code == 200
    assert result.attempts == 2
    assert calls == [
        "https://a.example/v1/chat/completions",
        "https://b.example/v1/chat/completions",
    ]
    assert len(records) == 2
    malformed, successful = records
    assert malformed.provider == "a"
    assert malformed.attempt == 1
    assert malformed.outcome is Outcome.FAILURE
    assert malformed.status_code == 200
    assert malformed.error_category is ErrorCategory.MALFORMED_RESPONSE
    assert malformed.fallback_attempted is True
    assert successful.provider == "b"
    assert successful.attempt == 2
    assert successful.outcome is Outcome.SUCCESS
    assert successful.status_code == 200
    assert successful.error_category is None
    assert successful.fallback_attempted is False
    assert malformed.trace_id == successful.trace_id
    assert json.loads(result.body)["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_invalid_json_mode_content_falls_back():
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "a.example" in str(request.url):
            return json_response(completion("not json"))
        return json_response(completion('{"ok": true}'))

    json_mode = caps(json_mode="supported")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(response_format={"type": "json_object"}),
            make_settings(
                provider("a", capabilities=json_mode),
                provider("b", capabilities=json_mode),
            ),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.attempts == 2
    assert calls == [
        "https://a.example/v1/chat/completions",
        "https://b.example/v1/chat/completions",
    ]
    assert len(records) == 2
    malformed, successful = records
    assert malformed.provider == "a"
    assert malformed.outcome is Outcome.FAILURE
    assert malformed.status_code == 200
    assert malformed.error_category is ErrorCategory.INVALID_JSON_CONTENT
    assert malformed.fallback_attempted is True
    assert successful.provider == "b"
    assert successful.outcome is Outcome.SUCCESS
    assert successful.error_category is None


@pytest.mark.asyncio
async def test_all_malformed_successes_return_generic_502_without_leaking_body(caplog):
    marker = "NEOROUTE_SECRET_MARKER_42"
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(
            200,
            content=marker.encode(),
            headers={"content-type": "application/json"},
        )

    with caplog.at_level(logging.INFO, logger="neoroute.gateway"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(HTTPException) as exc_info:
                await forward_chat(
                    make_payload(),
                    make_settings(provider("a"), provider("b")),
                    client=client,
                    attempt_log=records,
                )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == {
        "message": "All eligible providers failed",
        "attempts": [
            {"provider": "a", "error": "body_not_json"},
            {"provider": "b", "error": "body_not_json"},
        ],
    }
    assert marker not in str(exc_info.value.detail)
    assert marker not in repr(records)
    assert marker not in caplog.text
    assert len(calls) == 2
    assert len(records) == 2
    assert [record.provider for record in records] == ["a", "b"]
    assert all(
        record.outcome is Outcome.FAILURE for record in records
    )
    assert all(
        record.status_code == 200 for record in records
    )
    assert all(
        record.error_category is ErrorCategory.MALFORMED_RESPONSE
        for record in records
    )
    assert records[0].fallback_attempted is True
    assert records[1].fallback_attempted is False


@pytest.mark.asyncio
async def test_malformed_response_then_non_retryable_400_stops():
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "a.example" in str(request.url):
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(400, json={"error": "bad request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HTTPException) as exc_info:
            await forward_chat(
                make_payload(),
                make_settings(provider("a"), provider("b")),
                client=client,
                attempt_log=records,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["message"] == (
        "Upstream rejected the request; fallback stopped"
    )
    assert exc_info.value.detail["provider"] == "b"
    assert calls == [
        "https://a.example/v1/chat/completions",
        "https://b.example/v1/chat/completions",
    ]
    assert len(records) == 2
    malformed, rejected = records
    assert malformed.provider == "a"
    assert malformed.error_category is ErrorCategory.MALFORMED_RESPONSE
    assert malformed.status_code == 200
    assert malformed.fallback_attempted is True
    assert rejected.provider == "b"
    assert rejected.error_category is ErrorCategory.PROVIDER_CLIENT_ERROR
    assert rejected.status_code == 400
    assert rejected.fallback_attempted is False


@pytest.mark.parametrize(
    ("bad_body", "content_type"),
    [
        (b"", "application/json"),
        (b"<html>upstream failure</html>", "text/html"),
    ],
)
@pytest.mark.asyncio
async def test_empty_or_html_success_falls_back(bad_body, content_type):
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            return httpx.Response(
                200,
                content=bad_body,
                headers={"content-type": content_type},
            )
        return json_response(completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(),
            make_settings(provider("a"), provider("b")),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.attempts == 2
    assert records[0].provider == "a"
    assert records[0].error_category is ErrorCategory.MALFORMED_RESPONSE
    assert records[0].status_code == 200
    assert records[0].fallback_attempted is True
    assert records[1].outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_valid_tool_call_with_null_content_is_accepted():
    raw = json.dumps(
        completion(None, tool_calls=TOOL_CALLS)
    ).encode("utf-8")
    records: list[AttemptRecord] = []

    async def handler(_: httpx.Request):
        return httpx.Response(
            200,
            content=raw,
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(tools=TOOLS),
            make_settings(provider("a", capabilities=caps(tools="supported"))),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "a"
    assert result.attempts == 1
    assert result.body == raw
    assert len(records) == 1
    assert records[0].outcome is Outcome.SUCCESS
    assert records[0].error_category is None


@pytest.mark.asyncio
async def test_json_mode_empty_string_with_tool_calls_is_rejected():
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            return json_response(completion("", tool_calls=TOOL_CALLS))
        return json_response(completion('{"ok": true}'))

    combined = caps(json_mode="supported", tools="supported")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(
                response_format={"type": "json_object"},
                tools=TOOLS,
            ),
            make_settings(
                provider("a", capabilities=combined),
                provider("b", capabilities=combined),
            ),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.attempts == 2
    malformed, successful = records
    assert malformed.provider == "a"
    assert malformed.error_category is ErrorCategory.INVALID_JSON_CONTENT
    assert malformed.status_code == 200
    assert malformed.fallback_attempted is True
    assert successful.provider == "b"
    assert successful.outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_valid_response_bytes_content_type_and_status_are_preserved():
    raw = (
        b'{ "id":"chatcmpl-x", "choices": ['
        b'{"index":0,"message":{"role":"assistant","content":"h\xc3\xa9llo"},'
        b'"finish_reason":"stop"}],'
        b'"usage":{"total_tokens":3,"prompt_tokens":1,"completion_tokens":2},'
        b'"z_extra":{"b":2,"a":1} }'
    )
    content_type = "application/json; charset=utf-8"
    records: list[AttemptRecord] = []

    async def handler(_: httpx.Request):
        return httpx.Response(
            200,
            content=raw,
            headers={"content-type": content_type},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(),
            make_settings(provider("a")),
            client=client,
            attempt_log=records,
        )

    assert result.body == raw
    assert result.status_code == 200
    assert result.content_type == content_type
    assert result.provider_name == "a"
    assert result.attempts == 1
    assert records[0].outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_max_attempts_bounds_malformed_response_fallback():
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HTTPException) as exc_info:
            await forward_chat(
                make_payload(),
                make_settings(
                    provider("a"),
                    provider("b"),
                    provider("c"),
                    max_attempts=2,
                ),
                client=client,
                attempt_log=records,
            )

    assert exc_info.value.status_code == 502
    assert calls == [
        "https://a.example/v1/chat/completions",
        "https://b.example/v1/chat/completions",
    ]
    assert len(records) == 2
    assert [record.provider for record in records] == ["a", "b"]
    assert records[0].fallback_attempted is True
    assert records[1].fallback_attempted is False


@pytest.mark.asyncio
async def test_capability_filtering_is_separate_from_response_failures():
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "b.example" in str(request.url):
            return httpx.Response(200, content=b"not-json")
        return json_response(completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            image_payload(),
            make_settings(
                provider("a", capabilities=caps(vision="unsupported")),
                provider("b", capabilities=caps(vision="supported")),
                provider("c", capabilities=caps(vision="supported")),
            ),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "c"
    assert result.attempts == 2
    assert calls == [
        "https://b.example/v1/chat/completions",
        "https://c.example/v1/chat/completions",
    ]
    assert [record.provider for record in records] == ["b", "c"]
    assert records[0].error_category is ErrorCategory.MALFORMED_RESPONSE
    assert records[0].fallback_attempted is True
    assert records[1].outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_503_fallback_still_works():
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        if "a.example" in str(request.url):
            return httpx.Response(503, json={"error": "unavailable"})
        return json_response(completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(),
            make_settings(provider("a"), provider("b")),
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.attempts == 2
    assert records[0].error_category is ErrorCategory.PROVIDER_SERVER_ERROR
    assert records[0].fallback_attempted is True
    assert records[1].outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_400_still_stops_without_fallback():
    calls: list[str] = []
    records: list[AttemptRecord] = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(400, json={"error": "bad request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HTTPException) as exc_info:
            await forward_chat(
                make_payload(),
                make_settings(provider("a"), provider("b")),
                client=client,
                attempt_log=records,
            )

    assert exc_info.value.status_code == 400
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0].error_category is ErrorCategory.PROVIDER_CLIENT_ERROR
    assert records[0].fallback_attempted is False


class _ExplodingClient:
    def __init__(self, *args, **kwargs):
        raise AssertionError("HTTP client must not be constructed for streaming")


@pytest.mark.asyncio
async def test_streaming_still_returns_501_without_upstream_call():
    records: list[AttemptRecord] = []

    with (
        patch("neoroute.gateway.httpx.AsyncClient", _ExplodingClient),
        pytest.raises(HTTPException) as exc_info,
    ):
        await forward_chat(
            make_payload(stream=True),
            make_settings(provider("a")),
            attempt_log=records,
        )

    assert exc_info.value.status_code == 501
    assert records == []
