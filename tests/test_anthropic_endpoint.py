"""Route-level tests for the Anthropic /v1/messages translation surface.

Every provider here is a deterministic in-process mock (httpx.MockTransport).
No live provider is contacted, no real credentials are used, and no key value
ever appears in an assertion or response body.
"""
import json

import httpx
import pytest

from neoroute import gateway, main
from neoroute.gateway import UpstreamResult

PROVIDERS = json.dumps([
    {
        "name": "primary",
        "base_url": "https://primary.test/v1",
        "api_key_env": "TEST_PRIMARY_KEY",
        "model": "primary-model",
    },
    {
        "name": "secondary",
        "base_url": "https://secondary.test/v1",
        "api_key_env": "TEST_SECONDARY_KEY",
        "model": "secondary-model",
    },
])

OPENAI_TEXT_BODY = json.dumps({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "primary-model",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello from upstream."},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
}).encode()

OPENAI_TOOL_BODY = json.dumps({
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 0,
    "model": "primary-model",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_7",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            }],
        },
        "finish_reason": "tool_calls",
    }],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
}).encode()


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("NEOROUTE_PROVIDERS", PROVIDERS)
    monkeypatch.setenv("TEST_PRIMARY_KEY", "sk-mock-primary")
    monkeypatch.setenv("TEST_SECONDARY_KEY", "sk-mock-secondary")
    main.get_settings.cache_clear()
    yield
    main.get_settings.cache_clear()


async def post_messages(payload=None, body=None):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://neoroute.test") as client:
        if body is not None:
            return await client.post("/v1/messages", content=body,
                                     headers={"content-type": "application/json"})
        return await client.post("/v1/messages", json=payload)


@pytest.fixture
def stub_forward(monkeypatch):
    """Replace forward_chat with a capturing stub returning a canned result."""
    captured = {}

    async def fake(payload, settings, client=None, attempt_log=None):
        captured["payload"] = payload
        captured["settings"] = settings
        body = captured.get("body", OPENAI_TEXT_BODY)
        return UpstreamResult(body, captured.get("status", 200), "application/json", "primary", 1), 0.001

    monkeypatch.setattr(main, "forward_chat", fake)
    return captured


@pytest.fixture
def real_forward(monkeypatch):
    """Route the request through the real gateway with a mocked HTTP transport."""
    def install(handler):
        calls: list[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return handler(request)

        async def wrapper(payload, settings, client=None, attempt_log=None):
            mock_client = httpx.AsyncClient(transport=httpx.MockTransport(recording))
            try:
                return await gateway.forward_chat(payload, settings, client=mock_client,
                                                  attempt_log=attempt_log)
            finally:
                await mock_client.aclose()

        monkeypatch.setattr(main, "forward_chat", wrapper)
        return calls

    return install


async def test_text_request_translates_and_returns_anthropic_shape(stub_forward):
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-sonnet-4-5"
    assert body["content"] == [{"type": "text", "text": "Hello from upstream."}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert body["id"].startswith("msg_")
    assert response.headers["x-neoroute-provider"] == "primary"
    assert response.headers["x-neoroute-attempts"] == "1"

    upstream = stub_forward["payload"]
    assert upstream.messages[0] == {"role": "system", "content": "Be terse."}
    assert upstream.messages[1] == {"role": "user", "content": "Hi"}
    assert upstream.max_tokens == 256
    assert upstream.stream is False
    assert "sk-mock" not in response.text


async def test_tool_request_and_tool_response_translation(stub_forward):
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "tool_choice": {"type": "any"},
        "tools": [{
            "name": "get_weather",
            "description": "Look up weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "22C and sunny"},
            ]},
        ],
    })
    assert response.status_code == 200

    upstream = stub_forward["payload"]
    assert upstream.tools[0]["function"]["name"] == "get_weather"
    assert upstream.tools[0]["function"]["parameters"]["type"] == "object"
    assert upstream.tool_choice == "required"
    assistant = upstream.messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert upstream.messages[2] == {"role": "tool", "tool_call_id": "toolu_1", "content": "22C and sunny"}

    stub_forward["body"] = OPENAI_TOOL_BODY
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0] == {
        "type": "tool_use",
        "id": "call_7",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }


async def test_transient_primary_failure_falls_back_to_secondary(real_forward):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary.test":
            return httpx.Response(503, json={"error": {"message": "temporarily unavailable"}})
        return httpx.Response(200, json=json.loads(OPENAI_TEXT_BODY))

    calls = real_forward(handler)
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 200
    assert response.headers["x-neoroute-provider"] == "secondary"
    assert response.headers["x-neoroute-attempts"] == "2"
    assert response.json()["content"][0]["text"] == "Hello from upstream."
    assert [c.url.host for c in calls] == ["primary.test", "secondary.test"]
    assert "sk-mock" not in response.text


async def test_non_retryable_400_stops_with_anthropic_error(real_forward):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    calls = real_forward(handler)
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "fallback stopped" in body["error"]["message"]
    assert len(calls) == 1  # no blind retry of an invalid request
    assert "sk-mock" not in response.text


async def test_stream_true_rejected_with_explicit_blocker(stub_forward):
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "SSE" in body["error"]["message"]
    assert "Claude Code" in body["error"]["message"]
    assert "payload" not in stub_forward  # gateway never called


async def test_missing_max_tokens_rejected():
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 400
    assert "max_tokens" in response.json()["error"]["message"]


async def test_image_block_rejected_explicitly():
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
        ]}],
    })
    assert response.status_code == 400
    assert "image" in response.json()["error"]["message"]


async def test_invalid_json_body_rejected():
    response = await post_messages(body="{not json")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_invalid_configuration_returns_anthropic_shaped_503(monkeypatch):
    monkeypatch.delenv("NEOROUTE_PROVIDERS")
    main.get_settings.cache_clear()
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 503
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "NEOROUTE_PROVIDERS" in body["error"]["message"]


async def test_unusable_upstream_body_maps_to_502(stub_forward):
    stub_forward["body"] = b"<html>not json</html>"
    response = await post_messages({
        "model": "claude-sonnet-4-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
