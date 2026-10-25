"""Non-streaming response contract verification.

This module inspects upstream chat-completion bodies before the gateway treats
them as successful. It validates only the structure required to make a response
usable and preserves the original bytes for forwarding; optional metadata is not
type-checked or rewritten.

Structural validity is kept separate from semantic correctness. A response that
passes these checks may still be semantically wrong, and response formats that
require schema conformance (``json_schema``) are intentionally not evaluated
here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from neoroute.schemas import ChatCompletionRequest


class ResponseFailureReason(StrEnum):
    """Fixed, payload-free reasons a non-streaming response is unusable."""

    BODY_NOT_JSON = "body_not_json"
    TOP_LEVEL_NOT_OBJECT = "top_level_not_object"
    MISSING_CHOICES = "missing_choices"
    INVALID_CHOICES = "invalid_choices"
    EMPTY_CHOICES = "empty_choices"
    INVALID_CHOICE = "invalid_choice"
    MISSING_MESSAGE = "missing_message"
    INVALID_MESSAGE = "invalid_message"
    INVALID_ROLE = "invalid_role"
    MISSING_CONTENT = "missing_content"
    INVALID_CONTENT = "invalid_content"
    INVALID_TOOL_CALLS = "invalid_tool_calls"
    INVALID_FUNCTION_CALL = "invalid_function_call"
    INVALID_JSON_CONTENT = "invalid_json_content"


@dataclass(frozen=True)
class ResponseCheck:
    """Result of validating one upstream response body."""

    reason: ResponseFailureReason | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


def _reject_non_finite(constant: str) -> Any:
    """Reject NaN/Infinity, which Python accepts but RFC 8259 JSON does not."""
    raise ValueError(f"Non-standard JSON constant: {constant}")


def _parse_json(value: str | bytes) -> Any:
    return json.loads(value, parse_constant=_reject_non_finite)


def expects_json_object(payload: ChatCompletionRequest) -> bool:
    """Return True when the validated request asked for ``json_object`` mode."""
    response_format = payload.model_dump().get("response_format")
    return (
        isinstance(response_format, Mapping)
        and response_format.get("type") == "json_object"
    )


def _is_parseable_json(content: str) -> bool:
    try:
        _parse_json(content)
    except (ValueError, RecursionError):
        # Deeply nested input can exhaust the decoder's recursion budget.
        return False
    return True


def _check_message(
    message: Mapping[str, Any], expects_json_object: bool
) -> ResponseFailureReason | None:
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        return ResponseFailureReason.INVALID_ROLE

    if "content" not in message:
        return ResponseFailureReason.MISSING_CONTENT
    content = message["content"]
    if content is not None and not isinstance(content, str):
        return ResponseFailureReason.INVALID_CONTENT

    tool_calls = message.get("tool_calls")
    has_tool_calls = False
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            return ResponseFailureReason.INVALID_TOOL_CALLS
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                return ResponseFailureReason.INVALID_TOOL_CALLS
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                return ResponseFailureReason.INVALID_TOOL_CALLS
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                return ResponseFailureReason.INVALID_TOOL_CALLS
            if not isinstance(function.get("arguments"), str):
                return ResponseFailureReason.INVALID_TOOL_CALLS
        has_tool_calls = bool(tool_calls)

    function_call = message.get("function_call")
    has_function_call = False
    if function_call is not None:
        if not isinstance(function_call, Mapping):
            return ResponseFailureReason.INVALID_FUNCTION_CALL
        name = function_call.get("name")
        if not isinstance(name, str) or not name.strip():
            return ResponseFailureReason.INVALID_FUNCTION_CALL
        if not isinstance(function_call.get("arguments"), str):
            return ResponseFailureReason.INVALID_FUNCTION_CALL
        has_function_call = True

    refusal = message.get("refusal")
    has_refusal = isinstance(refusal, str) and bool(refusal.strip())

    if not expects_json_object:
        return None

    # Tool-call and refusal outcomes with null content have no assistant JSON
    # content to parse. A non-null content field is still checked.
    if content is None:
        if has_tool_calls or has_function_call or has_refusal:
            return None
        return ResponseFailureReason.INVALID_JSON_CONTENT

    if not _is_parseable_json(content):
        return ResponseFailureReason.INVALID_JSON_CONTENT
    return None


def check_response(body: bytes, *, expects_json_object: bool) -> ResponseCheck:
    """Validate a non-streaming chat-completion body without modifying it.

    The caller must forward the original ``body`` bytes when the check passes.
    """
    try:
        document = _parse_json(body)
    except (ValueError, RecursionError):
        return ResponseCheck(ResponseFailureReason.BODY_NOT_JSON)

    if not isinstance(document, Mapping):
        return ResponseCheck(ResponseFailureReason.TOP_LEVEL_NOT_OBJECT)

    if "choices" not in document:
        return ResponseCheck(ResponseFailureReason.MISSING_CHOICES)

    choices = document["choices"]
    if not isinstance(choices, list):
        return ResponseCheck(ResponseFailureReason.INVALID_CHOICES)
    if not choices:
        return ResponseCheck(ResponseFailureReason.EMPTY_CHOICES)

    for choice in choices:
        if not isinstance(choice, Mapping):
            return ResponseCheck(ResponseFailureReason.INVALID_CHOICE)
        if "message" not in choice:
            return ResponseCheck(ResponseFailureReason.MISSING_MESSAGE)
        message = choice["message"]
        if not isinstance(message, Mapping):
            return ResponseCheck(ResponseFailureReason.INVALID_MESSAGE)

        reason = _check_message(message, expects_json_object)
        if reason is not None:
            return ResponseCheck(reason)

    return ResponseCheck()
