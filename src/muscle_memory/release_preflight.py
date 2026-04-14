"""Release preflight dry-run checks for local use and CI."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from muscle_memory.release_artifacts import (
    discover_release_artifacts,
    verify_release_artifacts,
    write_release_checksums,
)
from muscle_memory.release_notes import extract_release_notes


def validate_release_benchmark(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("thresholds_passed"):
        failed = ", ".join(data.get("failed_thresholds", [])) or "unknown"
        raise ValueError(f"benchmark thresholds failed: {failed}")


def validate_release_metadata(version: str, repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]
    if project_version != version:
        raise ValueError(
            f"pyproject.toml version {project_version!r} does not match requested version {version!r}"
        )

    init_text = (repo_root / "src" / "muscle_memory" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([^"]+)"', init_text)
    if not match:
        raise ValueError("Could not find __version__ in src/muscle_memory/__init__.py")
    init_version = match.group(1)
    if init_version != version:
        raise ValueError(
            f"src/muscle_memory/__init__.py version {init_version!r} does not match requested version {version!r}"
        )

    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        raise ValueError(f"CHANGELOG.md is missing a ## [{version}] section")


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def distribution_paths(version: str, dist_dir: Path) -> list[Path]:
    artifacts = discover_release_artifacts(dist_dir, version)
    return [artifact.path for artifact in artifacts]


def run_release_preflight(version: str, repo_root: Path) -> None:
    validate_release_metadata(version, repo_root)
    validate_release_benchmark(repo_root / "benchmark-run.json")
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    extract_release_notes(version, changelog)

    with tempfile.TemporaryDirectory(prefix="mm-release-preflight-") as temp_dir:
        dist_dir = Path(temp_dir) / "dist"
        _run(["uv", "build", "--out-dir", str(dist_dir)], cwd=repo_root)
        _run(
            ["uvx", "--from", "twine", "twine", "check", *[str(path) for path in distribution_paths(version, dist_dir)]],
            cwd=repo_root,
        )
        verify_release_artifacts(version, dist_dir)
        write_release_checksums(version, dist_dir)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: release_preflight.py <version>", file=sys.stderr)
        return 1

    version = args[0].strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version):
        print(f"invalid version: {version!r}", file=sys.stderr)
        return 1

    try:
        run_release_preflight(version, Path.cwd())
    except (OSError, subprocess.CalledProcessError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Release preflight passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
