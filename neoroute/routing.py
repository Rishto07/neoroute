from dataclasses import dataclass
from neoroute.capabilities import CapabilityRequirements
from neoroute.config import ProviderConfig


@dataclass(frozen=True)
class RoutingPlan:
    providers: tuple[ProviderConfig, ...]
    requested_model: str


def build_plan(
    providers: tuple[ProviderConfig, ...],
    requested_model: str,
    max_attempts: int,
    *,
    requirements: CapabilityRequirements | None = None,
) -> RoutingPlan:
    """Filter by required capabilities, then prefer a named provider.

    Filtering happens before preference ordering and truncation so an
    incompatible provider never consumes an attempt slot.
    """
    required = requirements if requirements is not None else CapabilityRequirements()
    eligible = [p for p in providers if required.satisfied_by(p.capabilities)]

    if requested_model not in ("", "auto"):
        preferred = [p for p in eligible if p.name == requested_model]
        remaining = [p for p in eligible if p.name != requested_model]
        ordered = preferred + remaining
    else:
        ordered = eligible
    return RoutingPlan(tuple(ordered[:max_attempts]), requested_model)

