"""Route-level smoke tests for the core dashboard/API surface.

These endpoints previously had no test coverage. No live provider is
contacted and no saved user report is overwritten: the report path is
redirected to a tmp file for the 404 case.
"""
import json

import httpx
import pytest

from neoroute import main

PROVIDERS = json.dumps([{
    "name": "primary",
    "base_url": "https://primary.test/v1",
    "api_key_env": "TEST_SMOKE_KEY",
    "model": "primary-model",
}])


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("NEOROUTE_PROVIDERS", PROVIDERS)
    monkeypatch.setenv("TEST_SMOKE_KEY", "sk-mock-smoke")
    main.get_settings.cache_clear()
    yield
    main.get_settings.cache_clear()


async def call(path, method="GET", **kwargs):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://neoroute.test") as client:
        return await client.request(method, path, **kwargs)


async def test_root_redirects_to_dashboard():
    response = await call("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


async def test_dashboard_serves_index_html():
    response = await call("/ui/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


async def test_health_reports_configured_providers():
    response = await call("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["configured_providers"] == ["primary"]
    assert body["streaming"] is False
    assert "sk-mock" not in response.text


async def test_eval_report_404_when_no_saved_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_REPORT_PATH", tmp_path / "does_not_exist.json")
    response = await call("/api/eval/report")
    assert response.status_code == 404
    assert "No saved evaluation report" in response.json()["detail"]
