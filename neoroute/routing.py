from dataclasses import dataclass
from neoroute.config import ProviderConfig


@dataclass(frozen=True)
class RoutingPlan:
    providers: tuple[ProviderConfig, ...]
    requested_model: str


def build_plan(providers: tuple[ProviderConfig, ...], requested_model: str, max_attempts: int) -> RoutingPlan:
    """Prefer a named provider when requested; otherwise use configured order."""
    if requested_model not in ("", "auto"):
        preferred = [p for p in providers if p.name == requested_model]
        remaining = [p for p in providers if p.name != requested_model]
        ordered = preferred + remaining
    else:
        ordered = list(providers)
    return RoutingPlan(tuple(ordered[:max_attempts]), requested_model)

