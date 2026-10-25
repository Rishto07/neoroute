from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "auto"
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False

    @model_validator(mode="after")
    def validate_messages(self):
        for message in self.messages:
            if not isinstance(message, dict) or not message.get("role"):
                raise ValueError("Each message must be an object with a role")
        return self
