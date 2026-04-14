"""Release preflight dry-run checks for local use and CI."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
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
from muscle_memory.eval.benchmark import (
    _current_source_tree_sha256,
    run_benchmark,
    summarize_benchmark_artifact,
)
from muscle_memory.models import Scope
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


def _current_repo_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _current_worktree_state(repo_root: Path) -> tuple[bool | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    status = result.stdout
    return status == "", hashlib.sha256(status.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_run_matches_repo(
    data: dict[str, object],
    repo_root: Path,
    benchmark_path: Path,
) -> bool:
    repo_head = data.get("repo_head")
    repo_path = data.get("repo_root")
    benchmark_run_path = data.get("benchmark_path")
    benchmark_sha256 = data.get("benchmark_sha256")
    worktree_clean = data.get("worktree_clean")
    worktree_state = data.get("worktree_state")
    if (
        not isinstance(repo_head, str)
        or not isinstance(repo_path, str)
        or not isinstance(benchmark_run_path, str)
        or not isinstance(benchmark_sha256, str)
        or not isinstance(worktree_clean, bool)
        or not isinstance(worktree_state, str)
    ):
        return False

    current_head = _current_repo_head(repo_root)
    if current_head is None:
        return False

    current_worktree_clean, current_worktree_state = _current_worktree_state(repo_root)
    if current_worktree_state is None:
        return False
    if not benchmark_path.exists():
        return False

    return (
        Path(repo_path).expanduser().resolve() == repo_root.resolve()
        and repo_head == current_head
        and Path(benchmark_run_path).expanduser().resolve() == benchmark_path.resolve()
        and benchmark_sha256 == _file_sha256(benchmark_path)
        and worktree_clean == current_worktree_clean
        and worktree_state == current_worktree_state
    )


def _project_release_config(repo_root: Path) -> Config:
    cfg = Config.load(
        start_dir=repo_root,
        db_path=repo_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
    )
    cfg.project_root = repo_root
    return cfg


def _benchmark_artifact_matches_repo(data: dict[str, object], repo_root: Path) -> bool:
    repo_head = data.get("repo_head")
    source_tree_sha256 = data.get("source_tree_sha256")
    current_source_tree_sha256 = _current_source_tree_sha256(repo_root)
    current_repo_head = _current_repo_head(repo_root)
    if (
        not isinstance(repo_head, str)
        or not isinstance(source_tree_sha256, str)
        or current_source_tree_sha256 is None
        or current_repo_head is None
    ):
        return False
    return repo_head == current_repo_head and source_tree_sha256 == current_source_tree_sha256


def load_release_benchmark(repo_root: Path) -> dict[str, object]:
    cfg = _project_release_config(repo_root)
    benchmark_path = cfg.db_path.parent / "benchmark.json"
    benchmark_run_path = repo_root / "benchmark-run.json"
    if benchmark_run_path.exists():
        data = json.loads(benchmark_run_path.read_text(encoding="utf-8"))
        if _benchmark_run_matches_repo(data, repo_root, benchmark_path):
            return data

    if not benchmark_path.exists():
        raise ValueError(
            "missing benchmark results: "
            f"expected {benchmark_run_path} or {benchmark_path}. "
            "Run `mm eval build` first or provide benchmark-run.json."
        )

    raw_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not cfg.db_path.exists():
        if not _benchmark_artifact_matches_repo(raw_benchmark, repo_root):
            raise ValueError(
                f"benchmark artifact {benchmark_path} does not match the current source state"
            )
        return asdict(summarize_benchmark_artifact(benchmark_path))

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
