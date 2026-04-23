"""Tests for the curl install script."""

from __future__ import annotations

import hashlib
import http.server
import os
import socketserver
import subprocess
import threading
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"
ROOT = Path(__file__).resolve().parents[1]


def _serve(directory: Path) -> tuple[str, socketserver.TCPServer]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def _run_install(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(INSTALLER)],
        cwd=ROOT,
        env=os.environ | env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_install_script_downloads_and_installs_binary(tmp_path: Path) -> None:
    release_dir = tmp_path / "download" / "v0.11.0"
    release_dir.mkdir(parents=True)

    binary = release_dir / "mm-linux-x86_64"
    binary.write_text("#!/bin/sh\necho 'muscle-memory 0.11.0'\n", encoding="utf-8")
    binary.chmod(0o755)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    (release_dir / "SHA256SUMS").write_text(f"{digest}  mm-linux-x86_64\n", encoding="utf-8")

    base_url, server = _serve(tmp_path)
    install_dir = tmp_path / "bin"
    try:
        result = _run_install(
            {
                "MM_VERSION": "0.11.0",
                "MM_RELEASE_BASE_URL": base_url,
                "MM_INSTALL_DIR": str(install_dir),
                "MM_UNAME_S": "Linux",
                "MM_UNAME_M": "x86_64",
            }
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0
    installed = install_dir / "mm"
    assert installed.exists()

    version_result = subprocess.run(
        [str(installed), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version_result.stdout.strip() == "muscle-memory 0.11.0"


def test_install_script_rejects_unsupported_platform() -> None:
    result = _run_install({"MM_UNAME_S": "Darwin", "MM_UNAME_M": "x86_64"})

    assert result.returncode == 1
    assert "Unsupported platform" in result.stderr


def test_install_script_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    release_dir = tmp_path / "download" / "v0.11.0"
    release_dir.mkdir(parents=True)

    binary = release_dir / "mm-linux-x86_64"
    binary.write_text("#!/bin/sh\necho 'muscle-memory 0.11.0'\n", encoding="utf-8")
    binary.chmod(0o755)
    (release_dir / "SHA256SUMS").write_text(
        "0000000000000000000000000000000000000000000000000000000000000000  mm-linux-x86_64\n",
        encoding="utf-8",
    )

    base_url, server = _serve(tmp_path)
    install_dir = tmp_path / "bin"
    try:
        result = _run_install(
            {
                "MM_VERSION": "0.11.0",
                "MM_RELEASE_BASE_URL": base_url,
                "MM_INSTALL_DIR": str(install_dir),
                "MM_UNAME_S": "Linux",
                "MM_UNAME_M": "x86_64",
            }
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 1
    assert "Checksum verification failed" in result.stderr
    assert not (install_dir / "mm").exists()
