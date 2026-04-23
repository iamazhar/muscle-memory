"""Tests for release artifact verification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from muscle_memory.release_artifacts import (
    ArtifactSpec,
    assert_version_output,
    build_checksum_manifest,
    build_checksum_manifest_for_paths,
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

    with pytest.raises(
        ValueError, match=r"Expected exactly one wheel for version 0.8.0 in .+; found 0"
    ):
        discover_release_artifacts(tmp_path, "0.8.0")


def test_discover_release_artifacts_requires_exactly_one_sdist(tmp_path: Path) -> None:
    (tmp_path / "muscle_memory-0.8.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (tmp_path / "muscle_memory-0.8.0.post1.tar.gz").write_text("sdist-b", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Expected exactly one sdist for version 0.8.0 in .+; found 0"
    ):
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


def test_build_checksum_manifest_writes_sorted_sha256_lines(tmp_path: Path) -> None:
    wheel = tmp_path / "muscle_memory-0.8.0-py3-none-any.whl"
    sdist = tmp_path / "muscle_memory-0.8.0.tar.gz"
    wheel.write_bytes(b"wheel-bytes")
    sdist.write_bytes(b"sdist-bytes")

    manifest_path = build_checksum_manifest(
        [
            ArtifactSpec(kind="wheel", path=wheel),
            ArtifactSpec(kind="sdist", path=sdist),
        ],
        tmp_path,
    )

    assert manifest_path == tmp_path / "SHA256SUMS"
    assert manifest_path.read_text(encoding="utf-8").splitlines() == [
        "9ceb18f15662bb87e54af2f5953c0484d2ef76f5444d87913360b9ef87d7296d  muscle_memory-0.8.0-py3-none-any.whl",
        "3493dfe12f9879d916893954eb5c64591ab724bd752d2d79a7b55e15b2417239  muscle_memory-0.8.0.tar.gz",
    ]


def test_build_checksum_manifest_for_paths_supports_binary_and_installer_assets(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "muscle_memory-0.11.0-py3-none-any.whl"
    sdist = tmp_path / "muscle_memory-0.11.0.tar.gz"
    binary = tmp_path / "mm-linux-x86_64"
    installer = tmp_path / "install.sh"
    for path, text in (
        (wheel, "wheel"),
        (sdist, "sdist"),
        (binary, "binary"),
        (installer, "installer"),
    ):
        path.write_text(text, encoding="utf-8")

    manifest_path = build_checksum_manifest_for_paths(
        [wheel, sdist, binary, installer],
        tmp_path,
    )

    assert manifest_path == tmp_path / "SHA256SUMS"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "install.sh" in manifest_text
    assert "mm-linux-x86_64" in manifest_text
    assert "muscle_memory-0.11.0-py3-none-any.whl" in manifest_text
    assert "muscle_memory-0.11.0.tar.gz" in manifest_text
