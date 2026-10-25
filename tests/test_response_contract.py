import json

import pytest

from neoroute.response_contract import (
    ResponseFailureReason,
    check_response,
    expects_json_object,
)
from neoroute.schemas import ChatCompletionRequest


def body_with_message(message, **choice_extra):
    return {"choices": [{"message": message, **choice_extra}]}


def assistant_message(content="ok", **extra):
    return {"role": "assistant", "content": content, **extra}


def check(document, *, json_mode=False):
    return check_response(
        json.dumps(document).encode("utf-8"),
        expects_json_object=json_mode,
    )


def make_payload(**extra):
    return ChatCompletionRequest(
        model="auto",
        messages=[{"role": "user", "content": "hello"}],
        **extra,
    )


def test_minimal_valid_response():
    assert check(body_with_message(assistant_message())).ok


def test_full_response_metadata_is_accepted():
    document = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": assistant_message(),
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        "system_fingerprint": "fp_test",
    }
    assert check(document).ok


def test_tool_call_with_null_content_is_valid():
    message = assistant_message(
        None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"id": 1}'},
            }
        ],
    )
    assert check(body_with_message(message)).ok


def test_empty_tool_calls_are_structurally_valid():
    assert check(body_with_message(assistant_message(tool_calls=[]))).ok


def test_null_tool_calls_are_structurally_valid():
    assert check(body_with_message(assistant_message(tool_calls=None))).ok


def test_legacy_function_call_is_valid():
    message = assistant_message(
        None,
        function_call={"name": "lookup", "arguments": '{"id": 1}'},
    )
    assert check(body_with_message(message)).ok


def test_multiple_choices_are_validated():
    document = {
        "choices": [
            {"message": assistant_message("first")},
            {"message": assistant_message("second")},
        ]
    }
    assert check(document).ok


def test_unknown_provider_fields_are_preserved_and_accepted():
    message = assistant_message(
        "ok",
        provider_trace={"region": "test"},
    )
    assert check(body_with_message(message, provider_choice_field=True)).ok


def test_optional_metadata_is_not_type_checked():
    document = {
        "id": 5,
        "object": 42,
        "created": "not-a-timestamp",
        "model": [],
        "choices": [
            {
                "index": "zero",
                "message": assistant_message(),
                "finish_reason": 42,
            }
        ],
        "usage": "not-an-object",
    }
    assert check(document).ok


def test_every_choice_is_structurally_validated():
    document = {
        "choices": [
            {"message": assistant_message("first")},
            {"message": {"content": "missing role"}},
        ]
    }
    assert check(document).reason is ResponseFailureReason.INVALID_ROLE


def test_refusal_with_null_content_is_valid():
    message = assistant_message(None, refusal="I cannot help with that.")
    assert check(body_with_message(message)).ok


def test_audio_style_null_content_is_structurally_valid():
    message = assistant_message(None, audio={"id": "audio_1", "data": "..."})
    assert check(body_with_message(message)).ok


def test_null_content_without_tool_call_is_structurally_valid():
    assert check(body_with_message(assistant_message(None))).ok


def test_utf8_content_is_valid():
    document = body_with_message(assistant_message("héllo"))
    result = check_response(
        json.dumps(document, ensure_ascii=False).encode("utf-8"),
        expects_json_object=False,
    )
    assert result.ok


def test_surrounding_whitespace_is_valid():
    document = body_with_message(assistant_message())
    raw = b" \n" + json.dumps(document).encode("utf-8") + b"\n "
    assert check_response(raw, expects_json_object=False).ok


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ({}, ResponseFailureReason.MISSING_CHOICES),
        ({"choices": "nope"}, ResponseFailureReason.INVALID_CHOICES),
        ({"choices": []}, ResponseFailureReason.EMPTY_CHOICES),
        ({"choices": ["nope"]}, ResponseFailureReason.INVALID_CHOICE),
        ({"choices": [{}]}, ResponseFailureReason.MISSING_MESSAGE),
        (
            {"choices": [{"message": "nope"}]},
            ResponseFailureReason.INVALID_MESSAGE,
        ),
        (
            body_with_message({"content": "ok"}),
            ResponseFailureReason.INVALID_ROLE,
        ),
        (
            body_with_message({"role": 42, "content": "ok"}),
            ResponseFailureReason.INVALID_ROLE,
        ),
        (
            body_with_message({"role": "   ", "content": "ok"}),
            ResponseFailureReason.INVALID_ROLE,
        ),
        (
            body_with_message({"role": "assistant"}),
            ResponseFailureReason.MISSING_CONTENT,
        ),
        (
            body_with_message(assistant_message(42)),
            ResponseFailureReason.INVALID_CONTENT,
        ),
        (
            body_with_message(assistant_message({"nested": True})),
            ResponseFailureReason.INVALID_CONTENT,
        ),
        (
            body_with_message(assistant_message([1, 2])),
            ResponseFailureReason.INVALID_CONTENT,
        ),
        (
            body_with_message(assistant_message(None, tool_calls="nope")),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(assistant_message(None, tool_calls=["nope"])),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(assistant_message(None, tool_calls=[{}])),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(
                assistant_message(
                    None,
                    tool_calls=[{"function": {"name": "lookup"}}],
                )
            ),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(
                assistant_message(
                    None,
                    tool_calls=[
                        {
                            "function": {
                                "name": 42,
                                "arguments": "{}",
                            }
                        }
                    ],
                )
            ),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(
                assistant_message(
                    None,
                    tool_calls=[
                        {
                            "function": {
                                "name": "   ",
                                "arguments": "{}",
                            }
                        }
                    ],
                )
            ),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(
                assistant_message(
                    None,
                    tool_calls=[
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"id": 1},
                            }
                        }
                    ],
                )
            ),
            ResponseFailureReason.INVALID_TOOL_CALLS,
        ),
        (
            body_with_message(assistant_message(None, function_call="nope")),
            ResponseFailureReason.INVALID_FUNCTION_CALL,
        ),
        (
            body_with_message(assistant_message(None, function_call={})),
            ResponseFailureReason.INVALID_FUNCTION_CALL,
        ),
        (
            body_with_message(
                assistant_message(
                    None,
                    function_call={"name": "lookup", "arguments": 42},
                )
            ),
            ResponseFailureReason.INVALID_FUNCTION_CALL,
        ),
    ],
)
def test_structural_failures(document, reason):
    assert check(document).reason is reason


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not-json",
        b"<html>error</html>",
        b"{",
    ],
)
def test_non_json_body_is_rejected(raw):
    result = check_response(raw, expects_json_object=False)
    assert result.reason is ResponseFailureReason.BODY_NOT_JSON


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"null",
        b"42",
        b"true",
        b'"text"',
    ],
)
def test_non_object_top_level_is_rejected(raw):
    result = check_response(raw, expects_json_object=False)
    assert result.reason is ResponseFailureReason.TOP_LEVEL_NOT_OBJECT


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_rfc_json_constants_are_rejected(constant):
    raw = (
        b'{"choices": [{"message": {"role": "assistant", "content": "ok"}}], '
        b'"value": '
        + constant
        + b"}"
    )
    result = check_response(raw, expects_json_object=False)
    assert result.reason is ResponseFailureReason.BODY_NOT_JSON


def test_non_utf8_body_is_rejected():
    raw = b'{"choices": [{"message": {"role": "assistant", "content": "\xff"}}]}'
    result = check_response(raw, expects_json_object=False)
    assert result.reason is ResponseFailureReason.BODY_NOT_JSON


@pytest.mark.parametrize(
    "content",
    [
        '{"a": 1}',
        "[1, 2]",
        '"text"',
        "42",
        "true",
        "null",
        '  {"a": 1}  ',
    ],
)
def test_json_mode_accepts_any_parseable_json_value(content):
    result = check(body_with_message(assistant_message(content)), json_mode=True)
    assert result.ok


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"a": 1} trailing',
        "```json\n{\"a\": 1}\n```",
        '{"a": NaN}',
    ],
)
def test_json_mode_rejects_unparseable_content(content):
    result = check(body_with_message(assistant_message(content)), json_mode=True)
    assert result.reason is ResponseFailureReason.INVALID_JSON_CONTENT


def test_json_mode_rejects_null_content_without_tool_call_or_refusal():
    result = check(body_with_message(assistant_message(None)), json_mode=True)
    assert result.reason is ResponseFailureReason.INVALID_JSON_CONTENT


def test_json_mode_skips_content_check_for_tool_calls():
    message = assistant_message(
        None,
        tool_calls=[
            {
                "function": {
                    "name": "lookup",
                    "arguments": '{"id": 1}',
                }
            }
        ],
    )
    assert check(body_with_message(message), json_mode=True).ok


def test_json_mode_still_validates_non_null_content_for_tool_calls():
    message = assistant_message(
        "not json",
        tool_calls=[
            {
                "function": {
                    "name": "lookup",
                    "arguments": '{"id": 1}',
                }
            }
        ],
    )
    assert check(body_with_message(message), json_mode=True).reason is (
        ResponseFailureReason.INVALID_JSON_CONTENT
    )


def test_json_mode_accepts_valid_json_content_with_tool_calls():
    message = assistant_message(
        '{"note": "ok"}',
        tool_calls=[
            {
                "function": {
                    "name": "lookup",
                    "arguments": '{"id": 1}',
                }
            }
        ],
    )
    assert check(body_with_message(message), json_mode=True).ok


def test_json_mode_skips_content_check_for_legacy_function_call():
    message = assistant_message(
        None,
        function_call={"name": "lookup", "arguments": "{}"},
    )
    assert check(body_with_message(message), json_mode=True).ok


def test_json_mode_skips_content_check_for_refusal():
    message = assistant_message(None, refusal="I cannot help with that.")
    assert check(body_with_message(message), json_mode=True).ok


def test_json_mode_validates_every_choice():
    document = {
        "choices": [
            {"message": assistant_message('{"a": 1}')},
            {"message": assistant_message("not json")},
        ]
    }
    assert check(document, json_mode=True).reason is (
        ResponseFailureReason.INVALID_JSON_CONTENT
    )


def test_json_mode_accepts_multiple_valid_choices():
    document = {
        "choices": [
            {"message": assistant_message('{"a": 1}')},
            {"message": assistant_message("[1, 2]")},
        ]
    }
    assert check(document, json_mode=True).ok


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        ({"type": "json_object"}, True),
        ({"type": "json_schema", "json_schema": {}}, False),
        ({"type": "text"}, False),
        ("json_object", False),
        (None, False),
    ],
)
def test_expects_json_object(response_format, expected):
    payload = make_payload(response_format=response_format)
    assert expects_json_object(payload) is expected


def test_expects_json_object_is_false_when_absent():
    assert expects_json_object(make_payload()) is False


def test_json_schema_request_does_not_trigger_content_validation():
    payload = make_payload(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        }
    )
    result = check_response(
        json.dumps(body_with_message(assistant_message("not json"))).encode(),
        expects_json_object=expects_json_object(payload),
    )
    assert result.ok
