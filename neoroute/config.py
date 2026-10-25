
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")


@dataclass(frozen=True)
class Settings:
    providers: tuple[ProviderConfig, ...]
    timeout_seconds: float = 45.0
    max_attempts: int = 3


def _load_providers(raw: str) -> tuple[ProviderConfig, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "NEOROUTE_PROVIDERS must be a valid JSON array"
        ) from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            "NEOROUTE_PROVIDERS must be a non-empty JSON array"
        )

    providers: list[ProviderConfig] = []
    names: set[str] = set()

    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each provider must be a JSON object")

        try:
            provider = ProviderConfig(
                name=str(item["name"]).strip(),
                base_url=str(item["base_url"]).rstrip("/"),
                api_key_env=str(item["api_key_env"]).strip(),
                model=str(item["model"]).strip(),
            )
        except KeyError as exc:
            raise ValueError(
                f"Provider is missing required field: {exc.args[0]}"
            ) from exc

        if not all((
            provider.name,
            provider.base_url,
            provider.api_key_env,
            provider.model,
        )):
            raise ValueError(
                "Provider name, base_url, api_key_env, and model cannot be empty"
            )

        if not provider.base_url.startswith(
            ("https://", "http://127.0.0.1", "http://localhost")
        ):
            raise ValueError(
                f"Provider {provider.name!r} must use HTTPS (localhost is allowed)"
            )

        if provider.name in names:
            raise ValueError(f"Duplicate provider name: {provider.name}")

        names.add(provider.name)
        providers.append(provider)

    return tuple(providers)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = os.getenv("NEOROUTE_PROVIDERS", "")

    if not raw:
        raise ValueError(
            "Set NEOROUTE_PROVIDERS in your environment or .env file"
        )

    return Settings(
        providers=_load_providers(raw),
        timeout_seconds=float(
            os.getenv("NEOROUTE_TIMEOUT_SECONDS", "45")
        ),
        max_attempts=max(
            1, int(os.getenv("NEOROUTE_MAX_ATTEMPTS", "3"))
        ),
    )
