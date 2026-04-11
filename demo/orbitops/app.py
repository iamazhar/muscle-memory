"""Small standard-library web app for the OrbitOps demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    from .data import dashboard_payload
    from .ui import render_dashboard_page, render_home_page
except ImportError:  # pragma: no cover - convenience when running from the demo dir
    from data import dashboard_payload  # type: ignore[no-redef]
    from ui import render_dashboard_page, render_home_page  # type: ignore[no-redef]

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class OrbitOpsHandler(BaseHTTPRequestHandler):
    server_version = "OrbitOpsDemo/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib signature
        self._handle_request(send_body=False)

    def _handle_request(self, *, send_body: bool) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(render_home_page(), send_body=send_body)
            return
        if path == "/dashboard":
            self._send_html(render_dashboard_page(), send_body=send_body)
            return
        if path == "/api/dashboard.json":
            self._send_json(dashboard_payload(), send_body=send_body)
            return
        if path.startswith("/static/"):
            self._serve_static(path.removeprefix("/static/"), send_body=send_body)
            return
        self._send_html(
            "<h1>404</h1><p>OrbitOps could not find that page.</p>",
            HTTPStatus.NOT_FOUND,
            send_body=send_body,
        )

    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover - console nicety
        print(f"[orbitops] {self.address_string()} - " + fmt % args)

    def _send_html(
        self,
        body: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        send_body: bool = True,
    ) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _send_json(
        self,
        payload: dict,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        send_body: bool = True,
    ) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _serve_static(self, relative_path: str, *, send_body: bool = True) -> None:
        target = (STATIC_DIR / relative_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            self._send_html(
                "<h1>404</h1><p>Static asset not found.</p>",
                HTTPStatus.NOT_FOUND,
                send_body=send_body,
            )
            return
        content_type, _ = mimetypes.guess_type(target.name)
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)


def create_server(host: str = "127.0.0.1", port: int = 8008) -> ThreadingHTTPServer:
    """Create a local OrbitOps demo server."""
    return ThreadingHTTPServer((host, port), OrbitOpsHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OrbitOps demo app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port)
    host, port = server.server_address
    print(f"OrbitOps demo running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down OrbitOps demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
