"""Tests for release preflight dry-run checks."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.eval.benchmark import BenchmarkRunResult
from muscle_memory.eval.benchmark import build_benchmark
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall, Trajectory
from muscle_memory.release_preflight import (
    distribution_paths,
    load_release_benchmark,
    run_release_preflight,
    validate_release_benchmark,
    validate_release_metadata,
)


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


def _write_successful_benchmark_run(path: Path, *, repo_root: Path | None = None, repo_head: str | None = None) -> None:
    data = {"thresholds_passed": True, "failed_thresholds": []}
    if repo_root is not None:
        data["repo_root"] = str(repo_root)
    if repo_head is not None:
        data["repo_head"] = repo_head
    path.write_text(
        json.dumps(data) + "\n",
        encoding="utf-8",
    )


def _fake_release_build(command: list[str], cwd: Path) -> None:
    if command[:3] == ["uv", "build", "--out-dir"]:
        dist_dir = Path(command[3])
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "muscle_memory-0.8.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        (dist_dir / "muscle_memory-0.8.0.tar.gz").write_text("sdist", encoding="utf-8")


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


def test_validate_release_benchmark_rejects_failed_thresholds(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        """
        {
          "version": 1,
          "thresholds_passed": false,
          "entries": []
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark thresholds failed"):
        validate_release_benchmark(benchmark)


def test_run_release_preflight_uses_repo_benchmark_run_json(tmp_path: Path, monkeypatch) -> None:
    _write_repo_fixture(tmp_path)
    _write_successful_benchmark_run(
        tmp_path / "benchmark-run.json",
        repo_root=tmp_path,
        repo_head="abc123",
    )

    monkeypatch.setattr("muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123", raising=False)
    monkeypatch.setattr("muscle_memory.release_preflight._run", _fake_release_build)
    monkeypatch.setattr("muscle_memory.release_preflight.verify_release_artifacts", lambda version, dist_dir: None)
    monkeypatch.setattr("muscle_memory.release_preflight.write_release_checksums", lambda version, dist_dir: dist_dir / "SHA256SUMS")

    run_release_preflight("0.8.0", tmp_path)


def test_run_release_preflight_falls_back_to_recomputing_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _FakeEmbedder:
        def embed_one(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

    _write_repo_fixture(tmp_path)
    (tmp_path / ".git").mkdir()
    config = Config(
        db_path=tmp_path / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=tmp_path,
        embedding_dims=4,
    )
    store = Store(config.db_path, embedding_dims=4)
    skill = Skill(
        activation="When pytest fails",
        execution="1. Run `pytest`",
        termination="Tests pass",
    )
    store.add_skill(skill, embedding=[1.0, 0.0, 0.0, 0.0])
    store.add_episode(
        Episode(
            user_prompt="run tests",
            trajectory=Trajectory(
                tool_calls=[ToolCall(name="Bash", arguments={"command": "pytest"}, result="5 passed")]
            ),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
    )
    build_benchmark(
        store,
        embedder=_FakeEmbedder(),
        output_path=config.db_path.parent / "benchmark.json",
    )

    monkeypatch.setattr("muscle_memory.release_preflight.Config.load", lambda **kwargs: config)
    monkeypatch.setattr("muscle_memory.release_preflight.make_embedder", lambda cfg: _FakeEmbedder())
    monkeypatch.setattr("muscle_memory.release_preflight._run", _fake_release_build)
    monkeypatch.setattr("muscle_memory.release_preflight.verify_release_artifacts", lambda version, dist_dir: None)
    monkeypatch.setattr("muscle_memory.release_preflight.write_release_checksums", lambda version, dist_dir: dist_dir / "SHA256SUMS")

    run_release_preflight("0.8.0", tmp_path)


def test_load_release_benchmark_ignores_stale_repo_benchmark_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    _write_successful_benchmark_run(tmp_path / "benchmark-run.json")
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")

    expected = BenchmarkRunResult(
        total=1,
        avg_relevance=0.2,
        avg_adherence=0.2,
        false_positive_rate=1.0,
        execution_success_rate=0.0,
        promotion_precision=0.0,
        thresholds_passed=False,
        failed_thresholds=["avg_relevance"],
    )

    monkeypatch.setattr("muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123", raising=False)
    monkeypatch.setattr("muscle_memory.release_preflight.make_embedder", lambda cfg: object())
    monkeypatch.setattr("muscle_memory.release_preflight.run_benchmark", lambda store, path, embedder=None: expected)

    data = load_release_benchmark(tmp_path)

    assert data == asdict(expected)


def test_load_release_benchmark_recompute_ignores_mm_db_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")

    external_db = tmp_path / "external.db"
    monkeypatch.setenv("MM_DB_PATH", str(external_db))
    monkeypatch.setenv("MM_SCOPE", "global")

    captured: dict[str, object] = {}
    expected = BenchmarkRunResult(
        total=1,
        avg_relevance=0.9,
        avg_adherence=0.9,
        false_positive_rate=0.0,
        execution_success_rate=1.0,
        promotion_precision=1.0,
        thresholds_passed=True,
        failed_thresholds=[],
    )

    class _FakeStore:
        def __init__(self, db_path: Path, *, embedding_dims: int) -> None:
            captured["db_path"] = db_path
            captured["embedding_dims"] = embedding_dims

    monkeypatch.setattr("muscle_memory.release_preflight.Store", _FakeStore)
    monkeypatch.setattr("muscle_memory.release_preflight.make_embedder", lambda cfg: object())
    monkeypatch.setattr("muscle_memory.release_preflight.run_benchmark", lambda store, path, embedder=None: expected)

    data = load_release_benchmark(tmp_path)

    assert data == asdict(expected)
    assert captured["db_path"] == tmp_path / ".claude" / "mm.db"
