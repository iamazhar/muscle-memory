from __future__ import annotations

from demo.orbitops.check import run_checks
from demo.orbitops.data import dashboard_payload
from demo.orbitops.ui import render_dashboard_page, render_home_page


def test_demo_dashboard_payload_has_expected_shape() -> None:
    payload = dashboard_payload()
    assert payload["brand"] == "OrbitOps"
    assert payload["default_segment"] in payload["segments"]
    assert len(payload["metrics"]) == 4
    assert payload["average_health"] > 0


def test_demo_pages_render_key_sections() -> None:
    home = render_home_page()
    dashboard = render_dashboard_page()

    assert "Open live dashboard" in home
    assert "Dogfood prompts" in home
    assert "Live rollout board" in dashboard
    assert "segment-switcher__button" in dashboard


def test_demo_smoke_checks_pass() -> None:
    run_checks()
