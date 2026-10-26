"""Shared, side-effect-free harness for replaying the response corpus.

Both the M4 pytest replay and the M5 evaluation runner build gateway requests
from validated fixtures through these helpers. Every upstream call is served
by ``httpx.MockTransport``: no provider is contacted and no real credential is
read. API keys are synthetic ``test-key-*`` values registered in an isolated
environment patch.

Nothing here performs I/O at import time.
"""
from __future__ import annotations

import os
from typing import Self

import httpx
from corpus_loader import Case

from neoroute.attempts import ErrorCategory
from neoroute.capabilities import ModelCapabilities
from neoroute.config import ProviderConfig, Settings
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

SYNTHETIC_PROVIDER_NAMES = frozenset({"alpha", "beta", "gamma"})
LEAK_MARKER = "NEOROUTE_M4_SECRET"

# Categories that cannot arise from the response path. ``forward_chat`` catches
# only ``httpx.TransportError``, and ``classify_exception`` maps every such
# exception to TIMEOUT or CONNECTION_ERROR (covered in tests/test_attempts.py).
# Any other exception is never classified and propagates, so UNKNOWN is
# unreachable here. Manufacturing a fake case would assert behavior the
# gateway never has.
UNEXERCISED_CATEGORIES = {
    ErrorCategory.UNKNOWN: (
        "The gateway catches only httpx.TransportError, which maps to TIMEOUT "
        "or CONNECTION_ERROR; UNKNOWN cannot arise from a provider attempt."
    ),
}


class EnvPatch:
    """Minimal ``os.environ`` patcher with a monkeypatch-compatible surface.

    The M4 replay uses pytest's ``monkeypatch``; the standalone evaluator has
    no pytest fixture, so it passes this object instead. ``undo`` restores every
    touched variable to its original value (or removes it when it was unset).
    """

    def __init__(self) -> None:
        self._previous: dict[str, str | None] = {}

    def _remember(self, name: str) -> None:
        if name not in self._previous:
            self._previous[name] = os.environ.get(name)

    def setenv(self, name: str, value: str) -> None:
        self._remember(name)
        os.environ[name] = value

    def delenv(self, name: str, raising: bool = True) -> None:
        if raising and name not in os.environ:
            raise KeyError(name)
        self._remember(name)
        os.environ.pop(name, None)

    def undo(self) -> None:
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._previous.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.undo()


def build_settings(case: Case, patcher) -> Settings:
    """Build settings for one fixture using synthetic per-provider keys."""
    providers = []
    for spec in case.providers:
        env_name = f"TEST_M4_KEY_{spec.name.upper()}"
        if spec.missing_api_key:
            patcher.delenv(env_name, raising=False)
        else:
            patcher.setenv(env_name, f"test-key-{spec.name}")
        providers.append(
            ProviderConfig(
                name=spec.name,
                base_url=f"https://{spec.name}.example/v1",
                api_key_env=env_name,
                model=f"model-{spec.name}",
                capabilities=ModelCapabilities.from_mapping(spec.capabilities),
            )
        )
    return Settings(
        providers=tuple(providers),
        timeout_seconds=2,
        max_attempts=case.max_attempts,
    )


def build_payload(case: Case) -> ChatCompletionRequest:
    extra = {}
    if "response_format" in case.request:
        extra["response_format"] = case.request["response_format"]
    if case.request.get("tools"):
        extra["tools"] = TOOLS
    return ChatCompletionRequest(
        model="auto",
        messages=[{"role": "user", "content": "hello"}],
        **extra,
    )


def build_handler(case: Case, calls: list[str]):
    host_to_index = {
        f"{spec.name}.example": index
        for index, spec in enumerate(case.providers)
    }

    def handler(request: httpx.Request) -> httpx.Response:
        index = host_to_index.get(request.url.host)
        if index is None:
            raise AssertionError(f"unexpected upstream host: {request.url.host}")
        calls.append(case.providers[index].name)
        fixture = case.responses[index]
        if fixture.raise_kind == "timeout":
            raise httpx.ReadTimeout("fixture timeout", request=request)
        if fixture.raise_kind == "connect_error":
            raise httpx.ConnectError("fixture connection failure", request=request)
        return httpx.Response(
            fixture.status_code,
            content=fixture.body,
            headers={"content-type": fixture.content_type},
        )

    return handler
