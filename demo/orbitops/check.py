"""Local smoke checks for the OrbitOps demo app."""

from __future__ import annotations

import json
import threading
from urllib.request import urlopen

try:
    from .app import create_server
except ImportError:  # pragma: no cover - convenience when running from the demo dir
    from app import create_server  # type: ignore[no-redef]


def _read(url: str) -> tuple[int, str, str]:
    with urlopen(url) as response:
        body = response.read().decode("utf-8")
        content_type = response.headers.get("Content-Type", "")
        return response.status, body, content_type


def run_checks() -> None:
    """Start the demo on an ephemeral port and verify the core routes."""
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        home_status, home_body, _ = _read(f"{base_url}/")
        dashboard_status, dashboard_body, _ = _read(f"{base_url}/dashboard")
        api_status, api_body, api_type = _read(f"{base_url}/api/dashboard.json")
        css_status, css_body, _ = _read(f"{base_url}/static/styles.css")

        assert home_status == 200
        assert "OrbitOps" in home_body
        assert "Open live dashboard" in home_body

        assert dashboard_status == 200
        assert "Live rollout board" in dashboard_body
        assert "segment-switcher__button" in dashboard_body

        assert api_status == 200
        assert "application/json" in api_type
        payload = json.loads(api_body)
        assert payload["brand"] == "OrbitOps"
        assert payload["default_segment"] in payload["segments"]
        assert len(payload["metrics"]) == 4

        assert css_status == 200
        assert ".hero" in css_body
        assert ".dashboard-grid" in css_body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def main() -> None:
    run_checks()
    print("OrbitOps demo checks passed.")


if __name__ == "__main__":
    main()
