"""Anthropic Messages API translation for Neoroute.

Implements the non-streaming subset of ``POST /v1/messages`` by converting the
Anthropic request shape into the internal OpenAI-compatible
``ChatCompletionRequest``, forwarding it through the existing gateway
(fallback, attempt records, bounded retries, response contract), and
converting the upstream result back to the Anthropic Messages response shape.

Honest scope limits (verified, not aspirational):
- Non-streaming only. ``stream: true`` is rejected with ``400`` because the
  gateway has no SSE passthrough; real Claude Code requires SSE streaming and
  is therefore not yet supported end to end.
- Content block support is text, tool_use, and tool_result. Image blocks and
  unknown block types are rejected explicitly instead of being silently
  dropped.
- Unknown top-level parameters (for example ``thinking`` or beta fields) are
  accepted and ignored, matching standard gateway pass-through behavior.
- ``POST /v1/messages/count_tokens`` is not implemented.

Errors are returned in the Anthropic envelope:
``{"type": "error", "error": {"type": ..., "message": ...}}``.
No credentials, prompts, or completions are ever placed on these responses.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi.responses import JSONResponse

from neoroute.schemas import ChatCompletionRequest

RETRYABLE_NOTE = (
    "Streaming responses are not supported by this Neoroute MVP "
    "(the gateway has no SSE passthrough). Claude Code requires SSE streaming, "
    "so real Claude Code is not yet supported; send stream=false."
)

_ALLOWED_ROLES = ("user", "assistant")
_ALLOWED_TOOL_CHOICE = ("auto", "any", "tool", "none")
_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "function_call": "tool_use",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


class AnthropicRequestError(Exception):
    """A client-visible Anthropic error with a fixed message and status."""

    def __init__(self, message: str, status: int = 400, type_: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type_ or error_type_for(status)


def error_type_for(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 413:
        return "request_too_large"
    if status == 429:
        return "rate_limit_error"
    if status in (400, 422):
        return "invalid_request_error"
    return "api_error"


def error_body(message: str, type_: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": type_, "message": message}}


def error_response(message: str, status: int, type_: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=error_body(message, type_ or error_type_for(status)))


def gateway_error_message(exc_detail: Any) -> str:
    """Flatten a gateway HTTPException detail into a single readable message."""
    if isinstance(exc_detail, dict):
        message = exc_detail.get("message")
        if isinstance(message, str):
            return message
        try:
            return json.dumps(exc_detail, sort_keys=True)
        except (TypeError, ValueError):
            return str(exc_detail)
    return str(exc_detail)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnthropicRequestError(message)


def _text_of(blocks: list[Any], *, where: str) -> str:
    parts: list[str] = []
    for block in blocks:
        _require(isinstance(block, dict), f"{where}: content blocks must be objects")
        kind = block.get("type")
        if kind == "text":
            _require(isinstance(block.get("text"), str), f"{where}: text blocks need a string 'text'")
            parts.append(block["text"])
        else:
            raise AnthropicRequestError(
                f"{where}: unsupported content block type {kind!r}; this MVP supports text, tool_use, and tool_result"
            )
    return "".join(parts)


def _system_text(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return _text_of(system, where="system")
    raise AnthropicRequestError("system must be a string or a list of text blocks")


def _tool_result_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for block in value:
            _require(isinstance(block, dict), "tool_result content blocks must be objects")
            if block.get("type") != "text":
                raise AnthropicRequestError(
                    f"tool_result: unsupported content block type {block.get('type')!r}; only text is supported"
                )
        return _text_of(value, where="tool_result")
    raise AnthropicRequestError("tool_result content must be a string or a list of text blocks")


def _convert_message(message: Any) -> list[dict[str, Any]]:
    _require(isinstance(message, dict), "each message must be an object")
    role = message.get("role")
    _require(role in _ALLOWED_ROLES, f"messages: role must be one of {list(_ALLOWED_ROLES)}")
    content = message.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]
    _require(isinstance(content, list), "messages: content must be a string or a list of blocks")

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []

    for block in content:
        _require(isinstance(block, dict), f"{role} message: content blocks must be objects")
        kind = block.get("type")
        if kind == "text":
            _require(isinstance(block.get("text"), str), f"{role} message: text blocks need a string 'text'")
            text_parts.append(block["text"])
        elif kind == "tool_use":
            _require(role == "assistant", "tool_use blocks are only valid in assistant messages")
            _require(isinstance(block.get("id"), str) and block.get("id"), "tool_use: 'id' is required")
            _require(isinstance(block.get("name"), str) and block.get("name"), "tool_use: 'name' is required")
            tool_input = block.get("input", {})
            _require(isinstance(tool_input, dict), "tool_use: 'input' must be an object")
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(tool_input, separators=(",", ":")),
                },
            })
        elif kind == "tool_result":
            _require(role == "user", "tool_result blocks are only valid in user messages")
            _require(isinstance(block.get("tool_use_id"), str) and block.get("tool_use_id"),
                     "tool_result: 'tool_use_id' is required")
            out.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": _tool_result_text(block.get("content")),
            })
        elif kind == "image":
            raise AnthropicRequestError("image content blocks are not supported by this MVP")
        else:
            raise AnthropicRequestError(f"unsupported content block type {kind!r}")

    if role == "assistant":
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.insert(0, msg)
    elif text_parts:
        out.append({"role": "user", "content": "".join(text_parts)})
    elif not out:
        out.append({"role": "user", "content": ""})
    return out


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    _require(isinstance(tools, list), "tools must be a list")
    converted = []
    for tool in tools:
        _require(isinstance(tool, dict), "each tool must be an object")
        name = tool.get("name")
        _require(isinstance(name, str) and name, "tools: 'name' is required")
        schema = tool.get("input_schema")
        _require(isinstance(schema, dict), f"tool {name!r}: 'input_schema' must be an object")
        description = tool.get("description", "")
        _require(isinstance(description, str), f"tool {name!r}: 'description' must be a string")
        converted.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": schema},
        })
    return converted


def _convert_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    _require(isinstance(tool_choice, dict), "tool_choice must be an object")
    kind = tool_choice.get("type")
    _require(kind in _ALLOWED_TOOL_CHOICE, f"tool_choice.type must be one of {list(_ALLOWED_TOOL_CHOICE)}")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    name = tool_choice.get("name")
    _require(isinstance(name, str) and name, "tool_choice: 'name' is required for type 'tool'")
    return {"type": "function", "function": {"name": name}}


def to_chat_request(raw: dict[str, Any]) -> ChatCompletionRequest:
    """Translate an Anthropic Messages request into ChatCompletionRequest.

    Raises AnthropicRequestError (400) for anything this MVP cannot honor
    honestly; nothing is silently dropped except unknown top-level parameters.
    """
    stream = raw.get("stream", False)
    _require(isinstance(stream, bool), "stream must be a boolean")
    if stream:
        raise AnthropicRequestError(RETRYABLE_NOTE, 400)

    model = raw.get("model")
    _require(isinstance(model, str) and model, "model: field required and must be a non-empty string")

    max_tokens = raw.get("max_tokens")
    _require(type(max_tokens) is int and max_tokens > 0, "max_tokens: field required and must be a positive integer")

    messages = raw.get("messages")
    _require(isinstance(messages, list) and messages, "messages: field required and must be a non-empty list")

    chat_messages: list[dict[str, Any]] = []
    system = _system_text(raw.get("system"))
    if system is not None:
        chat_messages.append({"role": "system", "content": system})
    for message in messages:
        chat_messages.extend(_convert_message(message))

    payload: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "stream": False,
        "max_tokens": max_tokens,
    }

    for source_key, target_key in (("temperature", "temperature"), ("top_p", "top_p")):
        value = raw.get(source_key)
        if value is not None:
            _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                     f"{source_key} must be a number")
            payload[target_key] = value

    stop_sequences = raw.get("stop_sequences")
    if stop_sequences is not None:
        _require(isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences),
                 "stop_sequences must be a list of strings")
        payload["stop"] = stop_sequences

    tools = raw.get("tools")
    if tools is not None:
        payload["tools"] = _convert_tools(tools)

    tool_choice = _convert_tool_choice(raw.get("tool_choice"))
    if tool_choice is not None:
        _require(tools is not None, "tool_choice requires tools")
        payload["tool_choice"] = tool_choice

    return ChatCompletionRequest.model_validate(payload)


def anthropic_response_body(openai_body: bytes, requested_model: str) -> dict[str, Any]:
    """Convert a validated OpenAI chat-completion body into Anthropic shape.

    The gateway already enforced the response contract, so parse failures here
    mean the upstream body is unusable; a ValueError is raised and mapped to
    502 by the route.
    """
    try:
        data = json.loads(openai_body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Upstream response could not be parsed as JSON: {exc}") from exc
    if not isinstance(data, dict):
        # ValueError is the route's contract: it maps to a 502 api_error.
        raise ValueError("Upstream response is not a JSON object")  # noqa: TRY004

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Upstream response has no choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        # ValueError is the route's contract: it maps to a 502 api_error.
        raise ValueError("Upstream response has no message")  # noqa: TRY004

    content: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            arguments = function.get("arguments") or "{}"
            try:
                tool_input = json.loads(arguments)
            except (ValueError, TypeError):
                tool_input = {"_unparsed": str(arguments)}
            if not isinstance(tool_input, dict):
                tool_input = {"_unparsed": tool_input}
            content.append({
                "type": "tool_use",
                "id": call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function.get("name") or "",
                "input": tool_input,
            })
            stop_reason = "tool_use"

    if not content:
        content.append({"type": "text", "text": ""})
    if stop_reason != "tool_use":
        stop_reason = _FINISH_REASON_MAP.get(choice.get("finish_reason") or "stop", "end_turn")

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens if type(input_tokens) is int else 0,
            "output_tokens": output_tokens if type(output_tokens) is int else 0,
        },
    }
