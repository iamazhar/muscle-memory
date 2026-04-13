"""Tests for release artifact verification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from muscle_memory.release_artifacts import (
    ArtifactSpec,
    assert_version_output,
    discover_release_artifacts,
)


def test_discover_release_artifacts_returns_expected_specs(tmp_path: Path) -> None:
    (tmp_path / "muscle_memory-0.8.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (tmp_path / "muscle_memory-0.8.0.tar.gz").write_text("sdist", encoding="utf-8")

    artifacts = discover_release_artifacts(tmp_path, "0.8.0")

    assert artifacts == [
        ArtifactSpec(kind="wheel", path=tmp_path / "muscle_memory-0.8.0-py3-none-any.whl"),
        ArtifactSpec(kind="sdist", path=tmp_path / "muscle_memory-0.8.0.tar.gz"),
    ]


def test_discover_release_artifacts_requires_exactly_one_wheel(tmp_path: Path) -> None:
    (tmp_path / "muscle_memory-0.8.0.tar.gz").write_text("sdist", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected exactly one wheel for version 0.8.0 in .+; found 0"):
        discover_release_artifacts(tmp_path, "0.8.0")


def test_discover_release_artifacts_requires_exactly_one_sdist(tmp_path: Path) -> None:
    (tmp_path / "muscle_memory-0.8.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (tmp_path / "muscle_memory-0.8.0.post1.tar.gz").write_text("sdist-b", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected exactly one sdist for version 0.8.0 in .+; found 0"):
        discover_release_artifacts(tmp_path, "0.8.0")


def test_discover_release_artifacts_ignores_stale_other_version_files(tmp_path: Path) -> None:
    (tmp_path / "muscle_memory-0.7.9-py3-none-any.whl").write_text("old-wheel", encoding="utf-8")
    (tmp_path / "muscle_memory-0.7.9.tar.gz").write_text("old-sdist", encoding="utf-8")
    (tmp_path / "muscle_memory-0.8.0-py3-none-any.whl").write_text("new-wheel", encoding="utf-8")
    (tmp_path / "muscle_memory-0.8.0.tar.gz").write_text("new-sdist", encoding="utf-8")

    artifacts = discover_release_artifacts(tmp_path, "0.8.0")

    assert artifacts == [
        ArtifactSpec(kind="wheel", path=tmp_path / "muscle_memory-0.8.0-py3-none-any.whl"),
        ArtifactSpec(kind="sdist", path=tmp_path / "muscle_memory-0.8.0.tar.gz"),
    ]


def test_assert_version_output_accepts_cli_version_output() -> None:
    assert_version_output("muscle-memory 0.8.0\n", "0.8.0")
    assert_version_output("0.8.0\n", "0.8.0")


def test_assert_version_output_rejects_mismatched_version() -> None:
    with pytest.raises(ValueError, match=r"Expected exact version '0.8.0' in CLI output"):
        assert_version_output("muscle-memory 0.8.1\n", "0.8.0")


def test_assert_version_output_rejects_partial_matches() -> None:
    with pytest.raises(ValueError, match=r"Expected exact version '0.8.0' in CLI output"):
        assert_version_output("muscle-memory 0.8.0rc1\n", "0.8.0")
