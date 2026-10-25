"""Static, per-provider capability metadata and request requirements.

This module is intentionally independent of provider health, HTTP handling, and
response inspection. Capability metadata is operator-declared and is used only
to keep a provider out of a routing plan when the request explicitly requires a
capability that the provider cannot satisfy.

Unknown is a first-class state. A capability that is not declared is UNKNOWN,
and UNKNOWN never satisfies a requirement. Request shapes that cannot be
classified safely are reported as invalid instead of being treated as
text-only, so callers can fail closed before routing.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from neoroute.schemas import ChatCompletionRequest


class Capability(StrEnum):
    """Capabilities that can be required by a chat-completion request."""

    VISION = "vision"
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    JSON_SCHEMA = "json_schema"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"


class CapabilityState(StrEnum):
    """Whether a provider's support for a capability is known."""

    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class RequestShapeReason(StrEnum):
    """Fixed, payload-free reasons a request cannot be safely routed."""

    INVALID_CONTENT = "invalid_content"
    MISSING_CONTENT_TYPE = "missing_content_type"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    INVALID_TOOLS = "invalid_tools"
    INVALID_FUNCTIONS = "invalid_functions"
    INVALID_FUNCTION_CALL = "invalid_function_call"
    INVALID_TOOL_CHOICE = "invalid_tool_choice"
    INVALID_TOOL_CALLS = "invalid_tool_calls"
    INVALID_MESSAGE = "invalid_message"
    INVALID_RESPONSE_FORMAT = "invalid_response_format"
    UNSUPPORTED_RESPONSE_FORMAT = "unsupported_response_format"
    INVALID_MODALITIES = "invalid_modalities"
    UNSUPPORTED_MODALITY = "unsupported_modality"


IMAGE_CONTENT_TYPES = frozenset({"image_url", "image", "input_image"})
AUDIO_CONTENT_TYPES = frozenset({"input_audio", "audio"})
TEXT_CONTENT_TYPES = frozenset({"text", "input_text"})
MEDIA_CONTENT_FIELDS = frozenset(
    {
        "image_url",
        "image",
        "input_image",
        "input_audio",
        "audio",
        "file",
        "file_data",
        "file_id",
        "video_url",
    }
)
TOOL_CHOICE_VALUES = frozenset({"none", "auto", "required"})
MODALITY_VALUES = frozenset({"text", "audio"})


class InvalidRequestShape(ValueError):
    """Raised when a request shape cannot be classified safely."""

    def __init__(self, reason: RequestShapeReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True)
class ModelCapabilities:
    """Operator-declared capability metadata for one configured provider."""

    supported: frozenset[Capability] = frozenset()
    unsupported: frozenset[Capability] = frozenset()

    def __post_init__(self) -> None:
        supported = frozenset(self.supported)
        unsupported = frozenset(self.unsupported)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "unsupported", unsupported)

        overlap = supported & unsupported
        if overlap:
            names = ", ".join(sorted(capability.value for capability in overlap))
            raise ValueError(
                f"Capabilities cannot be both supported and unsupported: {names}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, str]) -> ModelCapabilities:
        """Parse the ``capabilities`` object from provider configuration.

        ``UNKNOWN`` is equivalent to omitting a capability. Unknown capability
        names and invalid states are rejected so a typo cannot silently weaken
        routing.
        """
        if not isinstance(raw, Mapping):
            raise TypeError("capabilities must be a mapping")

        supported: set[Capability] = set()
        unsupported: set[Capability] = set()

        for name, value in raw.items():
            try:
                capability = Capability(name)
            except ValueError as exc:
                raise ValueError(f"Unknown capability: {name!r}") from exc

            try:
                state = CapabilityState(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid state for capability {name!r}: {value!r}"
                ) from exc

            if state is CapabilityState.SUPPORTED:
                supported.add(capability)
            elif state is CapabilityState.UNSUPPORTED:
                unsupported.add(capability)
            # UNKNOWN is the same as omitting the declaration.

        return cls(frozenset(supported), frozenset(unsupported))

    def state(self, capability: Capability) -> CapabilityState:
        if capability in self.supported:
            return CapabilityState.SUPPORTED
        if capability in self.unsupported:
            return CapabilityState.UNSUPPORTED
        return CapabilityState.UNKNOWN

    def supports(self, capability: Capability) -> bool:
        """Return True only for an explicitly SUPPORTED capability."""
        return capability in self.supported

    def is_known(self, capability: Capability) -> bool:
        return self.state(capability) is not CapabilityState.UNKNOWN


@dataclass(frozen=True)
class CapabilityRequirements:
    """The capabilities a request needs from a provider."""

    required: frozenset[Capability] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", frozenset(self.required))

    def __bool__(self) -> bool:
        return bool(self.required)

    def satisfied_by(self, capabilities: ModelCapabilities) -> bool:
        return all(capabilities.supports(capability) for capability in self.required)

    def missing_from(self, capabilities: ModelCapabilities) -> frozenset[Capability]:
        return frozenset(
            capability
            for capability in self.required
            if not capabilities.supports(capability)
        )


@dataclass(frozen=True)
class RequestShape:
    """Typed classification of a request's capability-relevant structure."""

    requirements: CapabilityRequirements = field(
        default_factory=CapabilityRequirements
    )
    error: RequestShapeReason | None = None

    @property
    def valid(self) -> bool:
        return self.error is None

    @property
    def required(self) -> frozenset[Capability]:
        return self.requirements.required

    def __bool__(self) -> bool:
        return bool(self.requirements)


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _content_part_shape(part: Mapping) -> RequestShape:
    part_type = part.get("type")

    if part_type is None:
        has_media = any(name in part for name in MEDIA_CONTENT_FIELDS)
        if has_media or not isinstance(part.get("text"), str):
            return RequestShape(error=RequestShapeReason.MISSING_CONTENT_TYPE)
        return RequestShape()

    if not isinstance(part_type, str):
        return RequestShape(error=RequestShapeReason.INVALID_CONTENT)

    # Detection is based on the declared part type. A text part may carry
    # unrelated nested fields without implying vision support.
    if part_type in TEXT_CONTENT_TYPES:
        return RequestShape()
    if part_type in IMAGE_CONTENT_TYPES:
        return RequestShape(
            CapabilityRequirements(frozenset({Capability.VISION}))
        )
    if part_type in AUDIO_CONTENT_TYPES:
        return RequestShape(
            CapabilityRequirements(frozenset({Capability.AUDIO_INPUT}))
        )
    return RequestShape(error=RequestShapeReason.UNSUPPORTED_CONTENT_TYPE)


def _content_shape(content: object) -> RequestShape:
    if content is None or isinstance(content, str):
        return RequestShape()
    if isinstance(content, Mapping):
        return _content_part_shape(content)
    if _is_sequence(content):
        required: set[Capability] = set()
        for part in content:
            if not isinstance(part, Mapping):
                return RequestShape(error=RequestShapeReason.INVALID_CONTENT)
            part_shape = _content_part_shape(part)
            if part_shape.error is not None:
                return part_shape
            required.update(part_shape.required)
        return RequestShape(CapabilityRequirements(frozenset(required)))
    return RequestShape(error=RequestShapeReason.INVALID_CONTENT)


def analyze_request_shape(payload: ChatCompletionRequest) -> RequestShape:
    """Classify a request without calling a provider.

    Invalid or unrecognized shapes are returned as an invalid ``RequestShape``
    rather than an empty requirement set, so a caller cannot mistakenly treat
    them as text-only.
    """
    raw = payload.model_dump()
    required: set[Capability] = set()

    tools = raw.get("tools")
    if tools is not None:
        if not _is_sequence(tools) or any(
            not isinstance(item, Mapping) for item in tools
        ):
            return RequestShape(error=RequestShapeReason.INVALID_TOOLS)
        if tools:
            required.add(Capability.TOOLS)

    functions = raw.get("functions")
    functions_supplied = False
    if functions is not None:
        if not _is_sequence(functions) or any(
            not isinstance(item, Mapping) for item in functions
        ):
            return RequestShape(error=RequestShapeReason.INVALID_FUNCTIONS)
        functions_supplied = bool(functions)
    if functions_supplied:
        required.add(Capability.TOOLS)

    function_call = raw.get("function_call")
    if function_call is not None:
        if isinstance(function_call, str):
            if function_call in ("", "none"):
                pass
            elif function_call == "auto":
                if functions_supplied:
                    required.add(Capability.TOOLS)
            else:
                return RequestShape(error=RequestShapeReason.INVALID_FUNCTION_CALL)
        elif isinstance(function_call, Mapping):
            required.add(Capability.TOOLS)
        else:
            return RequestShape(error=RequestShapeReason.INVALID_FUNCTION_CALL)

    tool_choice = raw.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in TOOL_CHOICE_VALUES:
                return RequestShape(error=RequestShapeReason.INVALID_TOOL_CHOICE)
        elif not isinstance(tool_choice, Mapping):
            return RequestShape(error=RequestShapeReason.INVALID_TOOL_CHOICE)

    response_format = raw.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, Mapping):
            return RequestShape(error=RequestShapeReason.INVALID_RESPONSE_FORMAT)
        format_type = response_format.get("type")
        if not isinstance(format_type, str):
            return RequestShape(error=RequestShapeReason.INVALID_RESPONSE_FORMAT)
        if format_type == "json_object":
            required.add(Capability.JSON_MODE)
        elif format_type == "json_schema":
            required.add(Capability.JSON_SCHEMA)
        elif format_type != "text":
            return RequestShape(
                error=RequestShapeReason.UNSUPPORTED_RESPONSE_FORMAT
            )

    messages = raw.get("messages", [])
    if not _is_sequence(messages):
        return RequestShape(error=RequestShapeReason.INVALID_MESSAGE)
    for message in messages:
        if not isinstance(message, Mapping) or not message.get("role"):
            return RequestShape(error=RequestShapeReason.INVALID_MESSAGE)

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not _is_sequence(tool_calls) or any(
                not isinstance(item, Mapping) for item in tool_calls
            ):
                return RequestShape(error=RequestShapeReason.INVALID_TOOL_CALLS)
            if tool_calls:
                required.add(Capability.TOOLS)

        message_function_call = message.get("function_call")
        if message_function_call is not None:
            if isinstance(message_function_call, Mapping):
                required.add(Capability.TOOLS)
            elif isinstance(message_function_call, str):
                if message_function_call not in ("", "none"):
                    return RequestShape(
                        error=RequestShapeReason.INVALID_FUNCTION_CALL
                    )
            else:
                return RequestShape(
                    error=RequestShapeReason.INVALID_FUNCTION_CALL
                )

        if message.get("role") == "tool":
            required.add(Capability.TOOLS)

        content_shape = _content_shape(message.get("content"))
        if content_shape.error is not None:
            return content_shape
        required.update(content_shape.required)

    modalities = raw.get("modalities")
    if modalities is not None:
        if not _is_sequence(modalities):
            return RequestShape(error=RequestShapeReason.INVALID_MODALITIES)
        for modality in modalities:
            if not isinstance(modality, str):
                return RequestShape(error=RequestShapeReason.INVALID_MODALITIES)
            if modality not in MODALITY_VALUES:
                return RequestShape(error=RequestShapeReason.UNSUPPORTED_MODALITY)
            if modality == "audio":
                required.add(Capability.AUDIO_OUTPUT)

    return RequestShape(CapabilityRequirements(frozenset(required)))


def requirements_from_payload(payload: ChatCompletionRequest) -> CapabilityRequirements:
    """Return requirements, raising ``InvalidRequestShape`` when unresolvable."""
    shape = analyze_request_shape(payload)
    if shape.error is not None:
        raise InvalidRequestShape(shape.error)
    return shape.requirements
