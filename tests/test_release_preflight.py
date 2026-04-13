"""Tests for release preflight dry-run checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from muscle_memory.release_preflight import distribution_paths, validate_release_metadata


def test_distribution_paths_ignores_non_distribution_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    wheel = tmp_path / "muscle_memory-0.8.0-py3-none-any.whl"
    sdist = tmp_path / "muscle_memory-0.8.0.tar.gz"
    wheel.write_text("wheel", encoding="utf-8")
    sdist.write_text("sdist", encoding="utf-8")

    assert distribution_paths("0.8.0", tmp_path) == [wheel, sdist]


def _write_repo_fixture(tmp_path: Path, *, version: str = "0.8.0", changelog_version: str = "0.8.0") -> None:
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nname = \"muscle-memory\"\nversion = \"{version}\"\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "src" / "muscle_memory"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{changelog_version}]\n\n- Notes\n",
        encoding="utf-8",
    )


def test_validate_release_metadata_accepts_matching_files(tmp_path: Path) -> None:
    _write_repo_fixture(tmp_path)

    validate_release_metadata("0.8.0", tmp_path)


def test_validate_release_metadata_rejects_version_mismatch(tmp_path: Path) -> None:
    _write_repo_fixture(tmp_path, version="0.8.1", changelog_version="0.8.0")

    with pytest.raises(ValueError, match=r"pyproject.toml version '0.8.1' does not match requested version '0.8.0'"):
        validate_release_metadata("0.8.0", tmp_path)


def test_validate_release_metadata_requires_matching_changelog_section(tmp_path: Path) -> None:
    _write_repo_fixture(tmp_path, changelog_version="0.7.9")

    with pytest.raises(ValueError, match=r"CHANGELOG.md is missing a ## \[0.8.0\] section"):
        validate_release_metadata("0.8.0", tmp_path)
