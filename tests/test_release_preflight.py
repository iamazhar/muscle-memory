"""Tests for release preflight dry-run checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

import muscle_memory.release_preflight as release_preflight
from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.eval.benchmark import BenchmarkRunResult, build_benchmark
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall, Trajectory
from muscle_memory.release_preflight import (
    distribution_paths,
    load_release_benchmark,
    run_release_preflight,
    validate_release_benchmark,
    validate_release_metadata,
)

runner = CliRunner()


def test_distribution_paths_ignores_non_distribution_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    wheel = tmp_path / "muscle_memory-0.8.0-py3-none-any.whl"
    sdist = tmp_path / "muscle_memory-0.8.0.tar.gz"
    wheel.write_text("wheel", encoding="utf-8")
    sdist.write_text("sdist", encoding="utf-8")

    assert distribution_paths("0.8.0", tmp_path) == [wheel, sdist]


def _write_repo_fixture(
    tmp_path: Path, *, version: str = "0.8.0", changelog_version: str = "0.8.0"
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "muscle-memory"\nversion = "{version}"\n',
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


def _write_successful_benchmark_run(
    path: Path,
    *,
    repo_root: Path | None = None,
    repo_head: str | None = None,
    source_tree_sha256: str | None = None,
) -> None:
    data = {"thresholds_passed": True, "failed_thresholds": []}
    if repo_root is not None:
        data["repo_root"] = str(repo_root)
    if repo_head is not None:
        data["repo_head"] = repo_head
    if source_tree_sha256 is not None:
        data["source_tree_sha256"] = source_tree_sha256
    data.setdefault("worktree_clean", True)
    data.setdefault("worktree_state", hashlib.sha256(b"").hexdigest())
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

    with pytest.raises(
        ValueError, match=r"pyproject.toml version '0.8.1' does not match requested version '0.8.0'"
    ):
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


def test_current_worktree_state_ignores_benchmark_run_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        release_preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="?? benchmark-run.json\n M src/muscle_memory/release_preflight.py\n",
            stderr="",
        ),
    )

    clean, state = release_preflight._current_worktree_state(tmp_path)

    expected_status = " M src/muscle_memory/release_preflight.py\n"
    assert clean is False
    assert state == hashlib.sha256(expected_status.encode("utf-8")).hexdigest()


def test_run_release_preflight_uses_repo_benchmark_run_json(tmp_path: Path, monkeypatch) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    db_path = tmp_path / ".claude" / "mm.db"
    db_path.write_text("db\n", encoding="utf-8")
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("artifact\n", encoding="utf-8")
    _write_successful_benchmark_run(
        tmp_path / "benchmark-run.json",
        repo_root=tmp_path,
        repo_head="abc123",
        source_tree_sha256="tree-sha",
    )
    data = json.loads((tmp_path / "benchmark-run.json").read_text(encoding="utf-8"))
    data["benchmark_path"] = str(benchmark_path.resolve())
    data["benchmark_sha256"] = hashlib.sha256(b"artifact\n").hexdigest()
    data["db_path"] = str(db_path.resolve())
    data["db_sha256"] = hashlib.sha256(b"db\n").hexdigest()
    (tmp_path / "benchmark-run.json").write_text(json.dumps(data) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head",
        lambda repo_root: "abc123",
        raising=False,
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (True, hashlib.sha256(b"").hexdigest()),
        raising=False,
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root: "tree-sha",
        raising=False,
    )
    monkeypatch.setattr("muscle_memory.release_preflight._run", _fake_release_build)
    monkeypatch.setattr(
        "muscle_memory.release_preflight.verify_release_artifacts", lambda version, dist_dir: None
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight.write_release_checksums",
        lambda version, dist_dir: dist_dir / "SHA256SUMS",
    )

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
                tool_calls=[
                    ToolCall(name="Bash", arguments={"command": "pytest"}, result="5 passed")
                ]
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
    monkeypatch.setattr(
        "muscle_memory.release_preflight.make_embedder", lambda cfg: _FakeEmbedder()
    )
    monkeypatch.setattr("muscle_memory.release_preflight._run", _fake_release_build)
    monkeypatch.setattr(
        "muscle_memory.release_preflight.verify_release_artifacts", lambda version, dist_dir: None
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight.write_release_checksums",
        lambda version, dist_dir: dist_dir / "SHA256SUMS",
    )

    run_release_preflight("0.8.0", tmp_path)


def test_load_release_benchmark_ignores_stale_repo_benchmark_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    _write_successful_benchmark_run(
        tmp_path / "benchmark-run.json",
        repo_root=tmp_path,
        repo_head="abc123",
        source_tree_sha256="tree-sha",
    )
    data = json.loads((tmp_path / "benchmark-run.json").read_text(encoding="utf-8"))
    data["benchmark_path"] = str(benchmark_path.resolve())
    data["benchmark_sha256"] = hashlib.sha256(b"{}\n").hexdigest()
    (tmp_path / "benchmark-run.json").write_text(json.dumps(data) + "\n", encoding="utf-8")
    (tmp_path / ".claude" / "mm.db").touch()

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

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head",
        lambda repo_root: "abc123",
        raising=False,
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (False, "dirty-now"),
        raising=False,
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight.Store", lambda db_path, embedding_dims: object()
    )
    monkeypatch.setattr("muscle_memory.release_preflight.make_embedder", lambda cfg: object())
    monkeypatch.setattr(
        "muscle_memory.release_preflight.run_benchmark", lambda store, path, embedder=None: expected
    )

    data = load_release_benchmark(tmp_path)

    assert data == asdict(expected)


def test_load_release_benchmark_uses_frozen_artifact_without_project_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "tree-sha",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.9,
                        "adherence_score": 0.95,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight.run_benchmark",
        lambda *args, **kwargs: pytest.fail("unexpected recompute"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    data = load_release_benchmark(tmp_path)

    assert data["total"] == 1
    assert data["avg_relevance"] == pytest.approx(0.9)
    assert data["baseline_avg_relevance"] == pytest.approx(0.9)
    assert data["execution_success_rate"] == pytest.approx(1.0)
    assert data["thresholds_passed"] is True


def test_load_release_benchmark_rejects_stale_frozen_artifact_without_project_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "old-tree",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.9,
                        "adherence_score": 0.95,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "new-tree",
    )

    with pytest.raises(ValueError, match="does not match the current source state"):
        load_release_benchmark(tmp_path)


def test_load_release_benchmark_rejects_untracked_source_file_without_project_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "tracked-only-tree",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.9,
                        "adherence_score": 0.95,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "scratch.py").write_text("print('new helper')\n", encoding="utf-8")

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tracked-plus-untracked-tree",
    )

    with pytest.raises(ValueError, match="does not match the current source state"):
        load_release_benchmark(tmp_path)


def test_load_release_benchmark_rejects_stale_frozen_artifact_with_project_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    db_path = tmp_path / ".claude" / "mm.db"
    db_path.write_text("db\n", encoding="utf-8")
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "old-tree",
                "entries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "new-tree",
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight.run_benchmark",
        lambda *args, **kwargs: pytest.fail("unexpected recompute"),
    )

    with pytest.raises(ValueError, match="does not match the current source state"):
        load_release_benchmark(tmp_path)


def test_load_release_benchmark_recompute_ignores_mm_db_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / ".claude" / "mm.db").touch()

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
    monkeypatch.setattr(
        "muscle_memory.release_preflight.run_benchmark", lambda store, path, embedder=None: expected
    )

    data = load_release_benchmark(tmp_path)

    assert data == asdict(expected)
    assert captured["db_path"] == tmp_path / ".claude" / "mm.db"


def test_load_release_benchmark_accepts_mm_eval_run_json_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    db_path = tmp_path / ".claude" / "mm.db"
    db_path.write_text("db\n", encoding="utf-8")
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    config = Config(
        db_path=db_path,
        scope=Scope.PROJECT,
        project_root=tmp_path,
        embedding_dims=4,
    )
    benchmark_result = BenchmarkRunResult(
        total=2,
        avg_relevance=0.9,
        avg_adherence=0.95,
        baseline_avg_relevance=0.88,
        baseline_avg_adherence=0.93,
        false_positive_rate=0.0,
        execution_success_rate=1.0,
        promotion_precision=1.0,
        thresholds_passed=True,
        failed_thresholds=[],
    )

    with (
        monkeypatch.context() as cli_patch,
        monkeypatch.context() as preflight_patch,
    ):
        cli_patch.setattr("muscle_memory.cli._load_config", lambda scope=None: config)
        cli_patch.setattr("muscle_memory.cli._open_store", lambda cfg: object())
        cli_patch.setattr("muscle_memory.cli._current_repo_head", lambda repo_root: "abc123")
        cli_patch.setattr(
            "muscle_memory.cli._current_worktree_state", lambda repo_root: (True, "clean-state")
        )
        cli_patch.setattr(
            "muscle_memory.cli._current_source_tree_sha256",
            lambda repo_root, excluded_paths=None: "tree-sha",
        )
        cli_patch.setattr(
            "muscle_memory.cli._file_sha256",
            lambda path: "db-sha" if path == db_path else "bench-sha",
        )
        cli_patch.setattr("muscle_memory.cli.make_embedder", lambda cfg: object())
        cli_patch.setattr(
            "muscle_memory.eval.benchmark.run_benchmark",
            lambda store, path, embedder=None: benchmark_result,
        )
        result = runner.invoke(app, ["eval", "run", "--benchmark", str(benchmark_path), "--json"])

        assert result.exit_code == 0
        (tmp_path / "benchmark-run.json").write_text(result.output, encoding="utf-8")

        preflight_patch.setattr(
            "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
        )
        preflight_patch.setattr(
            "muscle_memory.release_preflight._current_worktree_state",
            lambda repo_root: (True, "clean-state"),
        )
        preflight_patch.setattr(
            "muscle_memory.release_preflight._current_source_tree_sha256",
            lambda repo_root, excluded_paths=None: "tree-sha",
        )
        preflight_patch.setattr(
            "muscle_memory.release_preflight._file_sha256",
            lambda path: "db-sha" if path == db_path else "bench-sha",
        )
        data = load_release_benchmark(tmp_path)

    assert data["repo_root"] == str(tmp_path.resolve())
    assert data["repo_head"] == "abc123"
    assert data["thresholds_passed"] is True
    assert data["benchmark_path"] == str(benchmark_path.resolve())
    assert data["benchmark_sha256"] == "bench-sha"
    assert data["db_path"] == str(db_path.resolve())
    assert data["db_sha256"] == "db-sha"
    assert data["source_tree_sha256"] == "tree-sha"


def test_load_release_benchmark_accepts_custom_repo_local_benchmark_artifact(
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
                tool_calls=[
                    ToolCall(name="Bash", arguments={"command": "pytest"}, result="5 passed")
                ]
            ),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
    )
    custom_benchmark_path = tmp_path / "artifacts" / "custom-benchmark.json"
    custom_benchmark_path.parent.mkdir()

    with monkeypatch.context() as build_patch:
        build_patch.setattr("muscle_memory.cli._load_config", lambda scope=None: config)
        build_patch.setattr("muscle_memory.cli._open_store", lambda cfg: store)
        build_patch.setattr("muscle_memory.cli.make_embedder", lambda cfg: _FakeEmbedder())
        build_patch.setattr(
            "muscle_memory.eval.benchmark.find_project_root", lambda start=None: tmp_path
        )
        build_patch.setattr(
            "muscle_memory.eval.benchmark._current_repo_head", lambda repo_root: "abc123"
        )
        build_patch.setattr(
            "muscle_memory.eval.benchmark._current_source_tree_sha256",
            lambda repo_root, excluded_paths=None: "tree-sha",
        )
        build_result = runner.invoke(
            app,
            ["eval", "build", "--output", str(custom_benchmark_path)],
        )

    assert build_result.exit_code == 0

    with monkeypatch.context() as run_patch:
        run_patch.setattr("muscle_memory.cli._load_config", lambda scope=None: config)
        run_patch.setattr("muscle_memory.cli._open_store", lambda cfg: store)
        run_patch.setattr("muscle_memory.cli.make_embedder", lambda cfg: _FakeEmbedder())
        run_patch.setattr("muscle_memory.cli._current_repo_head", lambda repo_root: "abc123")
        run_patch.setattr(
            "muscle_memory.cli._current_worktree_state", lambda repo_root: (True, "clean-state")
        )
        run_patch.setattr(
            "muscle_memory.cli._current_source_tree_sha256",
            lambda repo_root, excluded_paths=None: "tree-sha",
        )
        run_result = runner.invoke(
            app,
            ["eval", "run", "--benchmark", str(custom_benchmark_path), "--json"],
        )

    assert run_result.exit_code == 0
    (tmp_path / "benchmark-run.json").write_text(run_result.output, encoding="utf-8")

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (True, "clean-state"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    cached = load_release_benchmark(tmp_path)

    assert cached["thresholds_passed"] is True
    assert cached["benchmark_path"] == str(custom_benchmark_path.resolve())

    config.db_path.unlink()
    artifact = load_release_benchmark(tmp_path)

    assert artifact["thresholds_passed"] is True
    assert artifact["total"] == 1
    assert not (tmp_path / ".claude" / "benchmark.json").exists()


def test_load_release_benchmark_prefers_custom_artifact_over_default_when_cache_misses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    default_benchmark_path = tmp_path / ".claude" / "benchmark.json"
    default_benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "default-tree",
                "entries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    custom_benchmark_path = tmp_path / "artifacts" / "custom-benchmark.json"
    custom_benchmark_path.parent.mkdir()
    custom_benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "custom-tree",
                "entries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "benchmark-run.json").write_text(
        json.dumps(
            {
                "thresholds_passed": True,
                "failed_thresholds": [],
                "repo_root": str(tmp_path.resolve()),
                "repo_head": "abc123",
                "source_tree_sha256": "custom-tree",
                "worktree_clean": True,
                "worktree_state": "clean-state",
                "benchmark_path": str(custom_benchmark_path.resolve()),
                "benchmark_sha256": hashlib.sha256(custom_benchmark_path.read_bytes()).hexdigest(),
                "db_path": str((tmp_path / ".claude" / "mm.db").resolve()),
                "db_sha256": "missing-db",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )

    def _fake_source_tree(repo_root: Path, *, excluded_paths=None):
        if excluded_paths and custom_benchmark_path in excluded_paths:
            return "custom-tree"
        return "default-tree"

    captured: dict[str, Path] = {}

    def _fake_summarize(path: Path) -> BenchmarkRunResult:
        captured["path"] = path
        return BenchmarkRunResult(total=7, thresholds_passed=True, failed_thresholds=[])

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        _fake_source_tree,
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight.summarize_benchmark_artifact", _fake_summarize
    )

    data = load_release_benchmark(tmp_path)

    assert captured["path"] == custom_benchmark_path
    assert data["total"] == 7


def test_load_release_benchmark_ignores_stale_db_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    db_path = tmp_path / ".claude" / "mm.db"
    db_path.write_text("current-db\n", encoding="utf-8")
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "benchmark-run.json").write_text(
        json.dumps(
            {
                "thresholds_passed": True,
                "failed_thresholds": [],
                "repo_root": str(tmp_path.resolve()),
                "repo_head": "abc123",
                "worktree_clean": True,
                "worktree_state": "clean-state",
                "benchmark_path": str(benchmark_path.resolve()),
                "benchmark_sha256": "bench-sha",
                "db_path": str(db_path.resolve()),
                "db_sha256": "stale-db-sha",
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
    captured: dict[str, object] = {}

    class _FakeStore:
        def __init__(self, db_path: Path, *, embedding_dims: int) -> None:
            captured["db_path"] = db_path
            captured["embedding_dims"] = embedding_dims

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (True, "clean-state"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._file_sha256",
        lambda path: "current-db-sha" if path == db_path else "bench-sha",
    )
    monkeypatch.setattr("muscle_memory.release_preflight.Store", _FakeStore)
    monkeypatch.setattr("muscle_memory.release_preflight.make_embedder", lambda cfg: object())
    monkeypatch.setattr(
        "muscle_memory.release_preflight.run_benchmark",
        lambda store, path, embedder=None: expected,
    )

    data = load_release_benchmark(tmp_path)

    assert data == asdict(expected)
    assert captured["db_path"] == db_path


def test_load_release_benchmark_ignores_dirty_worktree_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "tree-sha",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.4,
                        "adherence_score": 0.4,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "benchmark-run.json").write_text(
        json.dumps(
            {
                "thresholds_passed": True,
                "failed_thresholds": [],
                "repo_root": str(tmp_path.resolve()),
                "repo_head": "abc123",
                "worktree_clean": True,
                "worktree_state": "clean-state",
                "benchmark_path": str(benchmark_path.resolve()),
                "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (False, "dirty-state"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    data = load_release_benchmark(tmp_path)

    assert data["thresholds_passed"] is False
    assert "avg_relevance" in data["failed_thresholds"]


def test_load_release_benchmark_ignores_changed_dirty_file_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    db_path = tmp_path / ".claude" / "mm.db"
    db_path.write_text("db\n", encoding="utf-8")
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "benchmark-run.json").write_text(
        json.dumps(
            {
                "thresholds_passed": True,
                "failed_thresholds": [],
                "repo_root": str(tmp_path.resolve()),
                "repo_head": "abc123",
                "source_tree_sha256": "old-dirty-tree",
                "worktree_clean": False,
                "worktree_state": "dirty-state",
                "benchmark_path": str(benchmark_path.resolve()),
                "benchmark_sha256": "bench-sha",
                "db_path": str(db_path.resolve()),
                "db_sha256": "db-sha",
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
    captured: dict[str, object] = {}

    class _FakeStore:
        def __init__(self, db_path: Path, *, embedding_dims: int) -> None:
            captured["db_path"] = db_path

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (False, "dirty-state"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "new-dirty-tree",
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._file_sha256",
        lambda path: "db-sha" if path == db_path else "bench-sha",
    )
    monkeypatch.setattr("muscle_memory.release_preflight.Store", _FakeStore)
    monkeypatch.setattr("muscle_memory.release_preflight.make_embedder", lambda cfg: object())
    monkeypatch.setattr(
        "muscle_memory.release_preflight.run_benchmark",
        lambda store, path, embedder=None: expected,
    )

    data = load_release_benchmark(tmp_path)

    assert data == asdict(expected)
    assert captured["db_path"] == db_path


def test_load_release_benchmark_rejects_mismatched_explicit_benchmark_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "tree-sha",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.4,
                        "adherence_score": 0.4,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "other-benchmark.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "other-tree",
                "entries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "benchmark-run.json").write_text(
        json.dumps(
            {
                "thresholds_passed": True,
                "failed_thresholds": [],
                "repo_root": str(tmp_path.resolve()),
                "repo_head": "abc123",
                "worktree_clean": True,
                "worktree_state": "clean-state",
                "benchmark_path": str((tmp_path / "other-benchmark.json").resolve()),
                "benchmark_sha256": hashlib.sha256(
                    (tmp_path / "other-benchmark.json").read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (True, "clean-state"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    with pytest.raises(ValueError, match="does not match the current source state"):
        load_release_benchmark(tmp_path)


def test_load_release_benchmark_accepts_matching_source_tree_with_different_repo_head(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "branch-head",
                "source_tree_sha256": "tree-sha",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.9,
                        "adherence_score": 0.9,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "merge-head"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    data = load_release_benchmark(tmp_path)

    assert data["thresholds_passed"] is True
    assert data["failed_thresholds"] == []


def test_run_release_benchmark_gate_uses_frozen_artifact_without_repo_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "branch-head",
                "source_tree_sha256": "tree-sha",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.9,
                        "adherence_score": 0.9,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "merge-head"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    data = release_preflight.run_release_benchmark_gate(tmp_path)

    assert data["thresholds_passed"] is True
    assert data["failed_thresholds"] == []


def test_load_release_benchmark_falls_back_from_mismatched_benchmark_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_repo_fixture(tmp_path)
    (tmp_path / ".claude").mkdir()
    benchmark_path = tmp_path / ".claude" / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_head": "abc123",
                "source_tree_sha256": "tree-sha",
                "entries": [
                    {
                        "skill_id": "skill1",
                        "skill_activation": "When pytest fails",
                        "episode_id": "ep1",
                        "user_prompt": "run tests",
                        "relevance_score": 0.4,
                        "adherence_score": 0.4,
                        "correctness_verdict": "correct",
                        "correctness_confidence": "human",
                        "outcome": "success",
                        "scored_at": "2026-04-13T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "benchmark-run.json").write_text(
        json.dumps(
            {
                "thresholds_passed": True,
                "failed_thresholds": [],
                "repo_root": str(tmp_path.resolve()),
                "repo_head": "abc123",
                "worktree_clean": True,
                "worktree_state": "clean-state",
                "benchmark_path": str(benchmark_path.resolve()),
                "benchmark_sha256": "wrong-sha",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_repo_head", lambda repo_root: "abc123"
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_worktree_state",
        lambda repo_root: (True, "clean-state"),
    )
    monkeypatch.setattr(
        "muscle_memory.release_preflight._current_source_tree_sha256",
        lambda repo_root, excluded_paths=None: "tree-sha",
    )

    data = load_release_benchmark(tmp_path)

    assert data["thresholds_passed"] is False
    assert "avg_relevance" in data["failed_thresholds"]
