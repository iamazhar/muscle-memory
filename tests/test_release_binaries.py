"""Tests for standalone release binary helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from muscle_memory.release_binaries import (
    BinaryTarget,
    binary_asset_name,
    pyinstaller_command,
    release_binary_targets,
    smoke_check_binary,
    verify_binary_version_output,
)


def test_release_binary_targets_match_supported_matrix() -> None:
    assert release_binary_targets() == [
        BinaryTarget(key="darwin-arm64", runner="macos-latest", asset_name="mm-darwin-arm64"),
        BinaryTarget(key="linux-x86_64", runner="ubuntu-latest", asset_name="mm-linux-x86_64"),
    ]


def test_binary_asset_name_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unsupported binary target"):
        binary_asset_name("windows-x86_64")


def test_verify_binary_version_output_accepts_mm_prefix() -> None:
    verify_binary_version_output("muscle-memory 0.11.0\n", "0.11.0")
    verify_binary_version_output("0.11.0\n", "0.11.0")


def test_verify_binary_version_output_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected exact version '0.11.0'"):
        verify_binary_version_output("muscle-memory 0.11.1\n", "0.11.0")


def test_pyinstaller_command_collects_prompt_package_data(tmp_path: Path) -> None:
    command = pyinstaller_command(
        "darwin-arm64",
        dist_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert command[0:3] == [sys.executable, "-m", "PyInstaller"]
    assert "--collect-data" in command
    assert "--hidden-import" in command
    assert "muscle_memory.prompts" in command
    assert "mm-darwin-arm64" in command
    assert (
        str(Path(__file__).resolve().parents[1] / "scripts" / "mm_binary_entrypoint.py") in command
    )


def test_smoke_check_binary_runs_version_command(tmp_path: Path) -> None:
    binary = tmp_path / "mm-linux-x86_64"
    binary.write_text("#!/bin/sh\necho 'muscle-memory 0.11.0'\n", encoding="utf-8")
    binary.chmod(0o755)

    smoke_check_binary(binary, "0.11.0")
