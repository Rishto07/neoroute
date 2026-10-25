from neoroute.config import ProviderConfig
from neoroute.routing import build_plan


def provider(name: str) -> ProviderConfig:
    return ProviderConfig(name, "https://example.com/v1", "KEY", "model-x")


def test_auto_uses_configured_order_and_attempt_limit():
    plan = build_plan((provider("a"), provider("b"), provider("c")), "auto", 2)
    assert [p.name for p in plan.providers] == ["a", "b"]


def test_named_provider_is_preferred():
    plan = build_plan((provider("a"), provider("b"), provider("c")), "b", 3)
    assert [p.name for p in plan.providers] == ["b", "a", "c"]

