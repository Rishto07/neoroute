import json
import os
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from neoroute.attempts import Outcome
from neoroute.capabilities import (
    AUDIO_CONTENT_TYPES,
    IMAGE_CONTENT_TYPES,
    Capability,
    CapabilityRequirements,
    CapabilityState,
    InvalidRequestShape,
    ModelCapabilities,
    RequestShapeReason,
    analyze_request_shape,
    requirements_from_payload,
)
from neoroute.config import ProviderConfig, Settings, _load_providers
from neoroute.gateway import forward_chat
from neoroute.routing import build_plan
from neoroute.schemas import ChatCompletionRequest

VISION = Capability.VISION
TOOLS = Capability.TOOLS
JSON_MODE = Capability.JSON_MODE
JSON_SCHEMA = Capability.JSON_SCHEMA
AUDIO_INPUT = Capability.AUDIO_INPUT
AUDIO_OUTPUT = Capability.AUDIO_OUTPUT

SUCCESS_BODY = {
    "id": "chatcmpl-test",
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
}


def caps(**declarations: str) -> ModelCapabilities:
    return ModelCapabilities.from_mapping(declarations)


def make_provider(name: str, capabilities: ModelCapabilities | None = None) -> ProviderConfig:
    return ProviderConfig(
        name,
        f"https://{name}.example/v1",
        f"TEST_KEY_{name.upper()}",
        f"model-{name}",
        capabilities if capabilities is not None else ModelCapabilities(),
    )


def make_settings(*providers: ProviderConfig, max_attempts: int = 2) -> Settings:
    for provider in providers:
        os.environ[provider.api_key_env] = "test-key"
    return Settings(tuple(providers), 2, max_attempts)


def make_payload(content: object = "hello", **extra: object) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto",
        messages=[{"role": "user", "content": content}],
        **extra,
    )


def image_payload() -> ChatCompletionRequest:
    return make_payload(
        [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]
    )


def assistant_payload(**extra: object) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto",
        messages=[{"role": "assistant", "content": None, **extra}],
    )


class _ExplodingClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("httpx.AsyncClient must not be constructed")


def test_missing_capability_is_unknown_not_supported():
    capabilities = ModelCapabilities()

    assert capabilities.state(VISION) is CapabilityState.UNKNOWN
    assert capabilities.supports(VISION) is False
    assert capabilities.is_known(VISION) is False


def test_supported_unsupported_and_unknown_round_trip():
    capabilities = ModelCapabilities.from_mapping(
        {
            "vision": "supported",
            "tools": "unsupported",
            "json_mode": "unknown",
        }
    )

    assert capabilities.state(VISION) is CapabilityState.SUPPORTED
    assert capabilities.supports(VISION) is True
    assert capabilities.state(TOOLS) is CapabilityState.UNSUPPORTED
    assert capabilities.supports(TOOLS) is False
    assert capabilities.state(JSON_MODE) is CapabilityState.UNKNOWN
    assert capabilities.is_known(JSON_MODE) is False


@pytest.mark.parametrize(
    "declarations",
    [
        {"visoin": "supported"},
        {"vision": "maybe"},
    ],
)
def test_invalid_capability_name_or_state_is_rejected(declarations):
    with pytest.raises(ValueError):
        ModelCapabilities.from_mapping(declarations)


def test_overlapping_supported_and_unsupported_is_rejected():
    with pytest.raises(ValueError):
        ModelCapabilities(frozenset({VISION}), frozenset({VISION}))


def test_text_only_request_has_no_requirements():
    requirements = requirements_from_payload(make_payload())

    assert requirements.required == frozenset()
    assert bool(requirements) is False


def test_analyze_request_shape_distinguishes_text_only_from_invalid():
    text_shape = analyze_request_shape(make_payload())
    invalid_payload = make_payload([{"type": "file", "file": {"file_data": "x"}}])
    invalid_shape = analyze_request_shape(invalid_payload)

    assert text_shape.valid is True
    assert text_shape.error is None
    assert text_shape.requirements.required == frozenset()
    assert invalid_shape.valid is False
    assert invalid_shape.error is RequestShapeReason.UNSUPPORTED_CONTENT_TYPE
    with pytest.raises(InvalidRequestShape) as exc_info:
        requirements_from_payload(invalid_payload)
    assert exc_info.value.reason is RequestShapeReason.UNSUPPORTED_CONTENT_TYPE


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (
            make_payload(
                [{"type": "image_url_v2", "image_url": {"url": "x"}}]
            ),
            RequestShapeReason.UNSUPPORTED_CONTENT_TYPE,
        ),
        (
            make_payload([{"type": "file", "file": {"file_data": "x"}}]),
            RequestShapeReason.UNSUPPORTED_CONTENT_TYPE,
        ),
        (
            make_payload([{"type": "video_url", "video_url": {"url": "x"}}]),
            RequestShapeReason.UNSUPPORTED_CONTENT_TYPE,
        ),
        (
            make_payload([{"image_url": {"url": "x"}}]),
            RequestShapeReason.MISSING_CONTENT_TYPE,
        ),
        (
            make_payload([{"input_audio": {"data": "x"}}]),
            RequestShapeReason.MISSING_CONTENT_TYPE,
        ),
        (make_payload([{}]), RequestShapeReason.MISSING_CONTENT_TYPE),
        (
            make_payload([{"type": "text", "text": "ok"}, "not-a-part"]),
            RequestShapeReason.INVALID_CONTENT,
        ),
        (
            make_payload([{"type": 7, "text": "ok"}]),
            RequestShapeReason.INVALID_CONTENT,
        ),
        (make_payload(42), RequestShapeReason.INVALID_CONTENT),
        (
            make_payload(response_format={"type": "json"}),
            RequestShapeReason.UNSUPPORTED_RESPONSE_FORMAT,
        ),
        (
            make_payload(response_format={"type": None}),
            RequestShapeReason.INVALID_RESPONSE_FORMAT,
        ),
        (
            make_payload(response_format={}),
            RequestShapeReason.INVALID_RESPONSE_FORMAT,
        ),
        (
            make_payload(response_format="json_object"),
            RequestShapeReason.INVALID_RESPONSE_FORMAT,
        ),
        (
            make_payload(modalities="audio"),
            RequestShapeReason.INVALID_MODALITIES,
        ),
        (
            make_payload(modalities=["text", 5]),
            RequestShapeReason.INVALID_MODALITIES,
        ),
        (
            make_payload(modalities=["speech"]),
            RequestShapeReason.UNSUPPORTED_MODALITY,
        ),
        (
            make_payload(tools="not-a-list"),
            RequestShapeReason.INVALID_TOOLS,
        ),
        (
            make_payload(tools={"type": "function"}),
            RequestShapeReason.INVALID_TOOLS,
        ),
        (
            make_payload(tools=["not-a-tool"]),
            RequestShapeReason.INVALID_TOOLS,
        ),
        (
            make_payload(functions="not-a-list"),
            RequestShapeReason.INVALID_FUNCTIONS,
        ),
        (
            make_payload(functions={"name": "lookup"}),
            RequestShapeReason.INVALID_FUNCTIONS,
        ),
        (
            make_payload(functions=["not-a-function"]),
            RequestShapeReason.INVALID_FUNCTIONS,
        ),
        (
            make_payload(function_call="required"),
            RequestShapeReason.INVALID_FUNCTION_CALL,
        ),
        (
            make_payload(function_call=[]),
            RequestShapeReason.INVALID_FUNCTION_CALL,
        ),
        (
            make_payload(tool_choice="sometimes"),
            RequestShapeReason.INVALID_TOOL_CHOICE,
        ),
        (
            make_payload(tool_choice=[]),
            RequestShapeReason.INVALID_TOOL_CHOICE,
        ),
        (
            assistant_payload(function_call="auto"),
            RequestShapeReason.INVALID_FUNCTION_CALL,
        ),
        (
            assistant_payload(function_call=[]),
            RequestShapeReason.INVALID_FUNCTION_CALL,
        ),
        (
            assistant_payload(tool_calls="bad"),
            RequestShapeReason.INVALID_TOOL_CALLS,
        ),
        (
            assistant_payload(tool_calls=["bad"]),
            RequestShapeReason.INVALID_TOOL_CALLS,
        ),
    ],
)
def test_unrecognized_or_malformed_shapes_fail_closed(payload, reason):
    shape = analyze_request_shape(payload)

    assert shape.valid is False
    assert shape.error is reason
    assert shape.requirements.required == frozenset()
    with pytest.raises(InvalidRequestShape):
        requirements_from_payload(payload)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            make_payload(
                [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ]
            ),
            {VISION},
        ),
        (
            make_payload(
                [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": "x"},
                ]
            ),
            {VISION},
        ),
        (
            make_payload(
                [{"type": "text", "text": "hi", "image_url": {"url": "x"}}]
            ),
            set(),
        ),
        (make_payload(tools=[]), set()),
        (make_payload(functions=[]), set()),
        (make_payload(tool_choice="auto"), set()),
        (make_payload(tool_choice="none"), set()),
        (
            make_payload(
                tool_choice={
                    "type": "function",
                    "function": {"name": "lookup"},
                }
            ),
            set(),
        ),
        (make_payload(response_format={"type": "text"}), set()),
        (make_payload(modalities=["text"]), set()),
        (make_payload(modalities=[]), set()),
        (make_payload(function_call="none"), set()),
        (make_payload(function_call=""), set()),
        (make_payload(function_call="auto"), set()),
        (make_payload(function_call="auto", functions=[]), set()),
        (
            make_payload(
                function_call="auto",
                functions=[{"name": "lookup"}],
            ),
            {TOOLS},
        ),
        (make_payload(function_call={"name": "lookup"}), {TOOLS}),
    ],
)
def test_recognized_shapes_derive_expected_requirements(payload, expected):
    shape = analyze_request_shape(payload)

    assert shape.valid is True
    assert shape.requirements.required == frozenset(expected)


def test_assistant_function_call_none_does_not_require_tools():
    shape = analyze_request_shape(assistant_payload(function_call="none"))

    assert shape.valid is True
    assert shape.requirements.required == frozenset()


def test_assistant_function_call_empty_string_does_not_require_tools():
    shape = analyze_request_shape(assistant_payload(function_call=""))

    assert shape.valid is True
    assert shape.requirements.required == frozenset()


def test_assistant_function_call_mapping_requires_tools():
    shape = analyze_request_shape(
        assistant_payload(
            function_call={"name": "lookup", "arguments": "{}"}
        )
    )

    assert shape.valid is True
    assert shape.requirements.required == frozenset({TOOLS})


def test_empty_tool_calls_does_not_require_tools():
    shape = analyze_request_shape(assistant_payload(tool_calls=[]))

    assert shape.valid is True
    assert shape.requirements.required == frozenset()


def test_supplied_functions_require_tools_even_when_choice_is_none():
    shape = analyze_request_shape(
        make_payload(
            functions=[{"name": "lookup"}],
            function_call="none",
        )
    )

    assert shape.valid is True
    assert shape.requirements.required == frozenset({TOOLS})


@pytest.mark.parametrize("part_type", sorted(IMAGE_CONTENT_TYPES))
def test_image_content_requires_vision(part_type):
    payload = make_payload([{"type": part_type, "image_url": {"url": "x"}}])

    assert requirements_from_payload(payload).required == frozenset({VISION})


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"tools": [{"type": "function", "function": {"name": "lookup"}}]}, {TOOLS}),
        ({"functions": [{"name": "lookup"}]}, {TOOLS}),
        ({"function_call": "auto", "functions": [{"name": "lookup"}]}, {TOOLS}),
        ({"response_format": {"type": "json_object"}}, {JSON_MODE}),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"type": "object"}},
                }
            },
            {JSON_SCHEMA},
        ),
    ],
)
def test_tools_and_response_formats_require_capabilities(extra, expected):
    requirements = requirements_from_payload(make_payload(**extra))

    assert requirements.required == frozenset(expected)


def test_tool_conversation_requires_tools():
    payload = ChatCompletionRequest(
        model="auto",
        messages=[
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ],
    )

    assert requirements_from_payload(payload).required == frozenset({TOOLS})


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            make_payload(
                [{"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}]
            ),
            {AUDIO_INPUT},
        ),
        (make_payload(modalities=["text", "audio"]), {AUDIO_OUTPUT}),
    ],
)
def test_audio_input_and_output_requirements(payload, expected):
    assert requirements_from_payload(payload).required == frozenset(expected)


@pytest.mark.parametrize("part_type", sorted(AUDIO_CONTENT_TYPES))
def test_audio_content_types_require_audio_input(part_type):
    payload = make_payload([{"type": part_type, "audio": {"data": "x"}}])

    shape = analyze_request_shape(payload)

    assert shape.valid is True
    assert shape.requirements.required == frozenset({AUDIO_INPUT})


def test_multiple_requirements_are_combined():
    payload = ChatCompletionRequest(
        model="auto",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                    {"type": "input_audio", "input_audio": {"data": "x"}},
                ],
            }
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
        modalities=["text", "audio"],
    )

    shape = analyze_request_shape(payload)

    assert shape.valid is True
    assert shape.requirements.required == frozenset(
        {VISION, AUDIO_INPUT, TOOLS, JSON_SCHEMA, AUDIO_OUTPUT}
    )


def test_multiple_requirements_require_every_capability():
    providers = (
        make_provider("a", caps(vision="supported", tools="supported")),
        make_provider(
            "b",
            caps(
                vision="supported",
                tools="supported",
                json_schema="supported",
            ),
        ),
        make_provider(
            "c",
            caps(
                vision="supported",
                tools="unsupported",
                json_schema="supported",
            ),
        ),
    )

    plan = build_plan(
        providers,
        "auto",
        10,
        requirements=CapabilityRequirements({VISION, TOOLS, JSON_SCHEMA}),
    )

    assert [p.name for p in plan.providers] == ["b"]


def test_text_only_plan_ignores_declared_capability_states():
    providers = (
        make_provider(
            "a",
            caps(
                vision="unsupported",
                tools="unsupported",
                json_mode="unsupported",
            ),
        ),
        make_provider("b", caps(vision="unknown", tools="unknown")),
    )

    plan = build_plan(
        providers,
        "auto",
        2,
        requirements=CapabilityRequirements(),
    )

    assert [p.name for p in plan.providers] == ["a", "b"]


def test_no_requirements_preserves_existing_plan():
    providers = (make_provider("a"), make_provider("b"), make_provider("c"))

    old_plan = build_plan(providers, "auto", 2)
    new_plan = build_plan(
        providers,
        "auto",
        2,
        requirements=CapabilityRequirements(),
    )
    named_old = build_plan(providers, "b", 2)
    named_new = build_plan(
        providers,
        "b",
        2,
        requirements=CapabilityRequirements(),
    )

    assert [p.name for p in new_plan.providers] == ["a", "b"]
    assert new_plan.providers == old_plan.providers
    assert named_new.providers == named_old.providers


def test_vision_requirement_excludes_unsupported_and_unknown_models():
    providers = (
        make_provider("a", caps(vision="unsupported")),
        make_provider("b", caps(vision="supported")),
        make_provider("c"),
    )

    plan = build_plan(
        providers,
        "auto",
        10,
        requirements=CapabilityRequirements({VISION}),
    )

    assert [p.name for p in plan.providers] == ["b"]


def test_filtering_happens_before_attempt_truncation():
    providers = (
        make_provider("a"),
        make_provider("b", caps(vision="supported")),
        make_provider("c", caps(vision="supported")),
    )

    plan = build_plan(
        providers,
        "auto",
        1,
        requirements=CapabilityRequirements({VISION}),
    )

    assert [p.name for p in plan.providers] == ["b"]


def test_named_incompatible_provider_is_excluded_but_fallback_remains():
    providers = (
        make_provider("a", caps(vision="unsupported")),
        make_provider("b", caps(vision="supported")),
    )

    plan = build_plan(
        providers,
        "a",
        2,
        requirements=CapabilityRequirements({VISION}),
    )

    assert [p.name for p in plan.providers] == ["b"]


def test_named_eligible_provider_remains_preferred_after_filtering():
    providers = (
        make_provider("a", caps(vision="supported")),
        make_provider("b", caps(vision="supported")),
        make_provider("c"),
    )

    plan = build_plan(
        providers,
        "b",
        2,
        requirements=CapabilityRequirements({VISION}),
    )

    assert [p.name for p in plan.providers] == ["b", "a"]


def test_json_schema_requirement_does_not_match_json_mode_only_model():
    providers = (
        make_provider("a", caps(json_mode="supported")),
        make_provider("b", caps(json_schema="supported")),
    )

    schema_plan = build_plan(
        providers,
        "auto",
        2,
        requirements=CapabilityRequirements({JSON_SCHEMA}),
    )
    mode_plan = build_plan(
        providers,
        "auto",
        2,
        requirements=CapabilityRequirements({JSON_MODE}),
    )

    assert [p.name for p in schema_plan.providers] == ["b"]
    assert [p.name for p in mode_plan.providers] == ["a"]


@pytest.mark.asyncio
async def test_all_ineligible_returns_422_without_http_calls():
    providers = (
        make_provider("a"),
        make_provider("b", caps(vision="unsupported")),
    )
    settings = make_settings(*providers)
    calls: list[str] = []
    records = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(200, json=SUCCESS_BODY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(HTTPException) as exc_info:
            await forward_chat(
                image_payload(),
                settings,
                client=client,
                attempt_log=records,
            )

    assert exc_info.value.status_code == 422
    assert calls == []
    assert records == []
    assert exc_info.value.detail["required"] == ["vision"]


@pytest.mark.asyncio
async def test_invalid_shape_returns_422_before_client_construction():
    settings = make_settings(make_provider("a", caps(vision="supported")))
    records = []
    payload = make_payload(
        [{"type": "image_url_v2", "image_url": {"url": "x"}}]
    )

    with (
        patch("neoroute.gateway.httpx.AsyncClient", _ExplodingClient),
        pytest.raises(HTTPException) as exc_info,
    ):
        await forward_chat(payload, settings, attempt_log=records)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == {
        "message": "Unsupported or malformed request shape",
        "reason": "unsupported_content_type",
    }
    assert records == []


@pytest.mark.asyncio
async def test_invalid_shape_422_does_not_echo_payload_contents():
    settings = make_settings(make_provider("a", caps(vision="supported")))
    records = []
    payload = make_payload(
        [
            {
                "type": "secret_image_format_xyz",
                "image_url": {"url": "https://secret.example/token"},
            }
        ]
    )

    with (
        patch("neoroute.gateway.httpx.AsyncClient", _ExplodingClient),
        pytest.raises(HTTPException) as exc_info,
    ):
        await forward_chat(payload, settings, attempt_log=records)

    rendered = str(exc_info.value.detail)
    assert "secret_image_format_xyz" not in rendered
    assert "secret.example" not in rendered
    assert records == []


@pytest.mark.asyncio
async def test_all_ineligible_returns_422_before_client_construction():
    providers = (
        make_provider("a", caps(vision="unsupported")),
        make_provider("b"),
    )
    settings = make_settings(*providers)
    records = []

    with (
        patch("neoroute.gateway.httpx.AsyncClient", _ExplodingClient),
        pytest.raises(HTTPException) as exc_info,
    ):
        await forward_chat(image_payload(), settings, attempt_log=records)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == {
        "message": "No configured provider satisfies the required capabilities",
        "required": ["vision"],
    }
    assert records == []


@pytest.mark.asyncio
async def test_text_only_routes_despite_unsupported_declarations():
    providers = (
        make_provider(
            "a",
            caps(
                vision="unsupported",
                tools="unsupported",
                audio_input="unknown",
            ),
        ),
    )
    settings = make_settings(*providers)
    calls: list[str] = []
    records = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(200, json=SUCCESS_BODY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            make_payload(),
            settings,
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "a"
    assert calls == ["https://a.example/v1/chat/completions"]
    assert len(records) == 1


@pytest.mark.asyncio
async def test_excluded_provider_is_not_an_attempt_record():
    providers = (
        make_provider("a", caps(vision="unsupported")),
        make_provider("b", caps(vision="supported")),
    )
    settings = make_settings(*providers)
    calls: list[str] = []
    records = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(200, json=SUCCESS_BODY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            image_payload(),
            settings,
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.attempts == 1
    assert calls == ["https://b.example/v1/chat/completions"]
    assert len(records) == 1
    assert records[0].provider == "b"
    assert records[0].outcome is Outcome.SUCCESS


@pytest.mark.asyncio
async def test_capability_filtering_composes_with_health_fallback():
    providers = (
        make_provider("a", caps(vision="supported")),
        make_provider("b", caps(vision="supported")),
    )
    settings = make_settings(*providers)
    calls: list[str] = []
    records = []

    async def handler(request: httpx.Request):
        calls.append(str(request.url))
        if "a.example" in str(request.url):
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=SUCCESS_BODY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result, _ = await forward_chat(
            image_payload(),
            settings,
            client=client,
            attempt_log=records,
        )

    assert result.provider_name == "b"
    assert result.attempts == 2
    assert calls == [
        "https://a.example/v1/chat/completions",
        "https://b.example/v1/chat/completions",
    ]
    assert [record.provider for record in records] == ["a", "b"]
    assert records[0].outcome is Outcome.FAILURE
    assert records[1].outcome is Outcome.SUCCESS


def test_config_without_capabilities_is_backward_compatible():
    raw = json.dumps(
        [
            {
                "name": "legacy",
                "base_url": "https://legacy.example/v1",
                "api_key_env": "LEGACY_KEY",
                "model": "legacy-model",
            }
        ]
    )

    providers = _load_providers(raw)

    assert providers[0].capabilities == ModelCapabilities()
    assert providers[0].capabilities.state(VISION) is CapabilityState.UNKNOWN
    plan = build_plan(providers, "auto", 3)
    assert [p.name for p in plan.providers] == ["legacy"]


def test_config_parses_capabilities_and_rejects_invalid_entries():
    valid = json.dumps(
        [
            {
                "name": "visionary",
                "base_url": "https://visionary.example/v1",
                "api_key_env": "VISIONARY_KEY",
                "model": "visionary-model",
                "capabilities": {
                    "vision": "supported",
                    "tools": "unsupported",
                    "json_mode": "unknown",
                },
            }
        ]
    )

    provider = _load_providers(valid)[0]
    assert provider.capabilities.state(VISION) is CapabilityState.SUPPORTED
    assert provider.capabilities.state(TOOLS) is CapabilityState.UNSUPPORTED
    assert provider.capabilities.state(JSON_MODE) is CapabilityState.UNKNOWN

    for invalid in (
        {"vision": "maybe"},
        {"visoin": "supported"},
        "not-an-object",
    ):
        raw = json.dumps(
            [
                {
                    "name": "bad",
                    "base_url": "https://bad.example/v1",
                    "api_key_env": "BAD_KEY",
                    "model": "bad-model",
                    "capabilities": invalid,
                }
            ]
        )
        with pytest.raises(ValueError, match="invalid capabilities"):
            _load_providers(raw)


def test_config_rejects_duplicate_provider_keys():
    raw = (
        '[{"name":"first","name":"second",'
        '"base_url":"https://p.example/v1","api_key_env":"KEY","model":"m"}]'
    )

    with pytest.raises(ValueError, match="duplicate JSON object keys"):
        _load_providers(raw)


def test_config_rejects_duplicate_capability_keys():
    raw = (
        '[{"name":"p","base_url":"https://p.example/v1",'
        '"api_key_env":"KEY","model":"m",'
        '"capabilities":{"vision":"supported","vision":"unsupported"}}]'
    )

    with pytest.raises(ValueError, match="duplicate JSON object keys"):
        _load_providers(raw)


@pytest.mark.parametrize("invalid", [None, [], True, 7, "not-an-object"])
def test_config_rejects_non_object_capabilities(invalid):
    raw = json.dumps(
        [
            {
                "name": "bad",
                "base_url": "https://bad.example/v1",
                "api_key_env": "BAD_KEY",
                "model": "bad-model",
                "capabilities": invalid,
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid capabilities"):
        _load_providers(raw)


def test_config_empty_capabilities_are_unknown():
    raw = json.dumps(
        [
            {
                "name": "p",
                "base_url": "https://p.example/v1",
                "api_key_env": "KEY",
                "model": "m",
                "capabilities": {},
            }
        ]
    )

    provider = _load_providers(raw)[0]

    assert provider.capabilities == ModelCapabilities()
    assert provider.capabilities.state(VISION) is CapabilityState.UNKNOWN


def test_every_capability_has_a_state_and_can_be_required():
    capabilities = ModelCapabilities.from_mapping(
        {
            capability.value: CapabilityState.SUPPORTED.value
            for capability in Capability
        }
    )

    assert all(capabilities.supports(capability) for capability in Capability)
    requirements = CapabilityRequirements(frozenset(Capability))
    assert requirements.satisfied_by(capabilities) is True
    assert requirements.missing_from(ModelCapabilities()) == frozenset(Capability)
