"""Release preflight dry-run checks for local use and CI."""

from __future__ import annotations

from dataclasses import asdict
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.embeddings import make_embedder
from muscle_memory.eval.benchmark import run_benchmark
from muscle_memory.release_artifacts import (
    discover_release_artifacts,
    verify_release_artifacts,
    write_release_checksums,
)
from muscle_memory.release_notes import extract_release_notes


def _validate_release_benchmark_data(data: dict[str, object]) -> None:
    if not data.get("thresholds_passed"):
        failed = ", ".join(data.get("failed_thresholds", [])) or "unknown"
        raise ValueError(f"benchmark thresholds failed: {failed}")


def validate_release_benchmark(path: Path) -> None:
    _validate_release_benchmark_data(json.loads(path.read_text(encoding="utf-8")))


def load_release_benchmark(repo_root: Path) -> dict[str, object]:
    benchmark_run_path = repo_root / "benchmark-run.json"
    if benchmark_run_path.exists():
        return json.loads(benchmark_run_path.read_text(encoding="utf-8"))

    cfg = Config.load(start_dir=repo_root)
    benchmark_path = cfg.db_path.parent / "benchmark.json"
    if not benchmark_path.exists():
        raise ValueError(
            "missing benchmark results: "
            f"expected {benchmark_run_path} or {benchmark_path}. "
            "Run `mm eval build` first or provide benchmark-run.json."
        )

    store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
    return asdict(run_benchmark(store, benchmark_path, embedder=make_embedder(cfg)))


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
    _validate_release_benchmark_data(load_release_benchmark(repo_root))
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
