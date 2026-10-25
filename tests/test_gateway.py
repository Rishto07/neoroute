import json
import os
import httpx
import pytest
from fastapi import HTTPException
from neoroute.config import ProviderConfig, Settings
from neoroute.gateway import forward_chat
from neoroute.schemas import ChatCompletionRequest


def make_settings():
    os.environ["TEST_KEY_A"] = "key-a"
    os.environ["TEST_KEY_B"] = "key-b"
    return Settings((ProviderConfig("a", "https://a.example/v1", "TEST_KEY_A", "model-a"), ProviderConfig("b", "https://b.example/v1", "TEST_KEY_B", "model-b")), 2, 2)


@pytest.mark.asyncio
async def test_falls_back_from_503_to_success():
    calls = []
    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "a.example" in str(request.url):
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"id": "chatcmpl-test", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        payload = ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "hello"}])
        result, _ = await forward_chat(payload, make_settings(), client=client)
    assert result.provider_name == "b"
    assert result.attempts == 2
    assert len(calls) == 2
    assert json.loads(result.body)["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_stops_on_non_transient_400():
    async def handler(_: httpx.Request):
        return httpx.Response(400, json={"error": "bad request"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        payload = ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "hello"}])
        with pytest.raises(HTTPException) as exc:
            await forward_chat(payload, make_settings(), client=client)
    assert exc.value.status_code == 400

