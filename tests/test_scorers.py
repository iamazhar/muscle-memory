"""Tests for automated playbook scorers."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.eval.benchmark import BenchmarkRunResult
from muscle_memory.eval.scorers import (
    AdherenceScore,
    _extract_step_tokens,
    _parse_execution_steps,
    score_adherence,
    score_correctness,
    score_relevance,
)
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall, Trajectory

runner = CliRunner()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "mm.db", embedding_dims=4)


def _make_skill(**kwargs) -> Skill:
    defaults = {
        "id": uuid.uuid4().hex[:16],
        "activation": "When X happens",
        "execution": "1. Run `pytest`\n2. Check the output\n3. Fix any failures",
        "termination": "Done when tests pass",
    }
    defaults.update(kwargs)
    return Skill(**defaults)


def _make_trajectory(commands: list[tuple[str, str | None]]) -> Trajectory:
    """Make a trajectory from (command, result) pairs."""
    calls = []
    for cmd, result in commands:
        tc = ToolCall(name="Bash", arguments={"command": cmd})
        if result and result.startswith("ERR:"):
            tc.error = result[4:]
        else:
            tc.result = result
        calls.append(tc)
    return Trajectory(tool_calls=calls)


# ── Relevance tests ────────────────────────────────────────────────


class TestRelevanceScorer:
    def test_stored_distance_used(self, store):
        ep = Episode(user_prompt="test", trajectory=Trajectory())
        result = score_relevance(store, ep, "any_id", stored_distance=0.5)
        assert result.method == "stored"
        assert result.l2_distance == 0.5
        assert result.score == pytest.approx(0.75)

    def test_zero_distance_is_perfect(self, store):
        result = score_relevance(
            store, Episode(user_prompt="x", trajectory=Trajectory()), "id", stored_distance=0.0
        )
        assert result.score == 1.0

    def test_max_distance_is_zero(self, store):
        result = score_relevance(
            store, Episode(user_prompt="x", trajectory=Trajectory()), "id", stored_distance=2.0
        )
        assert result.score == 0.0

    def test_no_embedder_returns_zero(self, store):
        ep = Episode(user_prompt="test", trajectory=Trajectory())
        result = score_relevance(store, ep, "missing_skill")
        assert result.score == 0.0
        assert result.method == "recomputed"


# ── Adherence tests ────────────────────────────────────────────────


class TestParseSteps:
    def test_numbered_list(self):
        steps = _parse_execution_steps("1. Do X\n2. Do Y\n3. Do Z")
        assert len(steps) == 3
        assert steps[0] == "Do X"

    def test_dash_list(self):
        steps = _parse_execution_steps("- First thing\n- Second thing")
        assert len(steps) == 2

    def test_empty(self):
        assert _parse_execution_steps("") == []

    def test_short_lines_skipped(self):
        steps = _parse_execution_steps("1. ok\n2. Do something meaningful")
        assert len(steps) == 1  # "ok" is too short (< 4 chars)


class TestExtractTokens:
    def test_backtick_commands(self):
        tokens = _extract_step_tokens("Run `pytest tests/ -q` to verify")
        assert "pytest" in tokens
        assert "pytest tests/ -q" in tokens

    def test_file_paths(self):
        tokens = _extract_step_tokens("Edit src/config.py to change the setting")
        assert "src/config.py" in tokens

    def test_tool_names(self):
        tokens = _extract_step_tokens("Use Grep to find the pattern")
        assert "Grep" in tokens

    def test_fallback_to_words(self):
        tokens = _extract_step_tokens("Verify the installation works correctly")
        assert len(tokens) > 0
        assert "Verify" in tokens


class TestAdherenceScorer:
    def test_all_steps_matched(self):
        skill = _make_skill(execution="1. Run `pytest`\n2. Run `git commit`")
        traj = _make_trajectory(
            [
                ("pytest tests/", "5 passed in 1.2s"),
                ("git commit -m 'fix'", "[main abc1234] fix"),
            ]
        )
        result = score_adherence(skill, traj)
        assert result.score == 1.0
        assert result.total_steps == 2
        assert len(result.matched_steps) == 2

    def test_no_steps_matched(self):
        skill = _make_skill(execution="1. Run `pytest`\n2. Run `git commit`")
        traj = _make_trajectory([("ls", "file.py")])
        result = score_adherence(skill, traj)
        assert result.score == 0.0
        assert len(result.unmatched_steps) == 2

    def test_partial_match(self):
        skill = _make_skill(execution="1. Run `pytest`\n2. Run `git commit`\n3. Run `git push`")
        traj = _make_trajectory(
            [
                ("pytest", "5 passed"),
                ("ls", "ok"),
            ]
        )
        result = score_adherence(skill, traj)
        assert abs(result.score - 1 / 3) < 0.01

    def test_single_generic_step(self):
        skill = _make_skill(execution="Verify the result")
        traj = _make_trajectory([("ls", "ok")])
        result = score_adherence(skill, traj)
        # Has 1 step, tokens like "Verify" and "result" won't match "ls"
        assert result.score == 0.0
        assert result.total_steps == 1

    def test_edit_tool_detected(self):
        skill = _make_skill(execution="1. Edit `config.py` to update the setting")
        traj = Trajectory(
            tool_calls=[
                ToolCall(name="Edit", arguments={"file_path": "/path/to/config.py"}, result="ok"),
            ]
        )
        result = score_adherence(skill, traj)
        assert result.score == 1.0


# ── Correctness tests ──────────────────────────────────────────────


class TestCorrectnessScorer:
    def test_followed_and_succeeded(self):
        adh = AdherenceScore(
            score=0.8, total_steps=3, matched_steps=["a", "b"], unmatched_steps=["c"]
        )
        result = score_correctness(adh, Outcome.SUCCESS)
        assert result.verdict == "correct"
        assert result.confidence == "auto"

    def test_followed_and_failed(self):
        adh = AdherenceScore(score=0.6, total_steps=2)
        result = score_correctness(adh, Outcome.FAILURE)
        assert result.verdict == "incorrect"

    def test_ignored_and_succeeded(self):
        adh = AdherenceScore(score=0.2, total_steps=5)
        result = score_correctness(adh, Outcome.SUCCESS)
        assert result.verdict == "needs_review"

    def test_unknown_outcome(self):
        adh = AdherenceScore(score=0.9, total_steps=2)
        result = score_correctness(adh, Outcome.UNKNOWN)
        assert result.verdict == "needs_review"


# ── Benchmark tests ────────────────────────────────────────────────


class TestBenchmark:
    def test_build_creates_file(self, store, tmp_path):
        from muscle_memory.eval.benchmark import build_benchmark

        # Add a skill and episode
        skill = _make_skill(execution="1. Run `pytest`")
        store.add_skill(skill)
        ep = Episode(
            user_prompt="run tests",
            trajectory=_make_trajectory([("pytest", "5 passed in 1.2s")]),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
        store.add_episode(ep)

        entries, path = build_benchmark(store, output_path=tmp_path / "bench.json")
        assert len(entries) == 1
        assert path.exists()

        # Verify JSON structure
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert len(data["entries"]) == 1
        assert data["entries"][0]["skill_id"] == skill.id

    def test_build_records_source_tree_provenance(self, store, tmp_path, monkeypatch):
        from muscle_memory.eval.benchmark import build_benchmark

        skill = _make_skill(execution="1. Run `pytest`")
        store.add_skill(skill)
        episode = Episode(
            user_prompt="run tests",
            trajectory=_make_trajectory([("pytest", "5 passed in 1.2s")]),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
        store.add_episode(episode)

        monkeypatch.setattr("muscle_memory.eval.benchmark.find_project_root", lambda start=None: tmp_path)
        monkeypatch.setattr("muscle_memory.eval.benchmark._current_repo_head", lambda repo_root: "abc123")
        monkeypatch.setattr(
            "muscle_memory.eval.benchmark._current_source_tree_sha256",
            lambda repo_root: "tree-sha",
        )

        _, path = build_benchmark(store, output_path=tmp_path / "bench.json")
        data = json.loads(path.read_text())

        assert data["repo_head"] == "abc123"
        assert data["source_tree_sha256"] == "tree-sha"

    def test_run_detects_no_drift(self, store, tmp_path):
        from muscle_memory.eval.benchmark import build_benchmark, run_benchmark

        skill = _make_skill(execution="1. Run `pytest`")
        store.add_skill(skill)
        ep = Episode(
            user_prompt="run tests",
            trajectory=_make_trajectory([("pytest", "5 passed in 1.2s")]),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
        store.add_episode(ep)

        _, path = build_benchmark(store, output_path=tmp_path / "bench.json")
        result = run_benchmark(store, path)

        assert result.total == 1
        assert not result.improved
        assert not result.degraded

    def test_run_reports_release_gate_status_from_benchmark_path(self, store, tmp_path):
        from muscle_memory.eval.benchmark import build_benchmark, run_benchmark

        class _FakeEmbedder:
            def embed_one(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0, 0.0]

        skill = _make_skill(execution="1. Run `pytest`")
        store.add_skill(skill, embedding=[1.0, 0.0, 0.0, 0.0])
        ep = Episode(
            user_prompt="run tests",
            trajectory=_make_trajectory([("pytest", "5 passed in 1.2s")]),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
        store.add_episode(ep)

        _, path = build_benchmark(store, output_path=tmp_path / "bench.json")
        result = run_benchmark(store, path, embedder=_FakeEmbedder())

        assert result.total == 1
        assert result.avg_relevance == pytest.approx(1.0)
        assert result.avg_adherence == pytest.approx(1.0)
        assert result.execution_success_rate == pytest.approx(1.0)
        assert result.promotion_precision == pytest.approx(1.0)
        assert result.thresholds_passed is True
        assert result.failed_thresholds == []

    def test_run_uses_frozen_correctness_labels_for_gate_metrics(self, store, tmp_path):
        from muscle_memory.eval.benchmark import run_benchmark

        class _FakeEmbedder:
            def embed_one(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0, 0.0]

        skill = _make_skill(execution="1. Run `pytest`")
        store.add_skill(skill, embedding=[1.0, 0.0, 0.0, 0.0])
        episode = Episode(
            user_prompt="run tests",
            trajectory=_make_trajectory([("pytest", "5 passed in 1.2s")]),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
        store.add_episode(episode)

        benchmark_path = tmp_path / "bench.json"
        benchmark_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "skill_id": skill.id,
                            "skill_activation": skill.activation,
                            "episode_id": episode.id,
                            "user_prompt": episode.user_prompt,
                            "relevance_score": 1.0,
                            "adherence_score": 1.0,
                            "correctness_verdict": "incorrect",
                            "correctness_confidence": "human",
                            "outcome": episode.outcome.value,
                        }
                    ],
                }
            )
            + "\n"
        )

        result = run_benchmark(store, benchmark_path, embedder=_FakeEmbedder())

        assert result.false_positive_rate == pytest.approx(1.0)
        assert result.execution_success_rate == pytest.approx(0.0)
        assert result.promotion_precision == pytest.approx(0.0)
        assert "false_positive_rate" in result.failed_thresholds
        assert "promotion_precision" in result.failed_thresholds

    def test_run_does_not_count_failed_execution_as_success(self, store, tmp_path):
        from muscle_memory.eval.benchmark import build_benchmark, run_benchmark

        skill = _make_skill(execution="1. Run `pytest`")
        store.add_skill(skill)
        episode = Episode(
            user_prompt="run tests",
            trajectory=_make_trajectory([("pytest", "1 failed in 0.8s")]),
            outcome=Outcome.FAILURE,
            activated_skills=[skill.id],
        )
        store.add_episode(episode)

        _, benchmark_path = build_benchmark(store, output_path=tmp_path / "bench.json")
        result = run_benchmark(store, benchmark_path)

        assert result.execution_success_rate == pytest.approx(0.0)
        assert "execution_success_rate" in result.failed_thresholds


@pytest.mark.parametrize(
    ("thresholds_passed", "expected_exit_code", "failed_thresholds"),
    [
        (True, 0, []),
        (False, 1, ["avg_relevance"]),
    ],
)
def test_eval_run_json_reports_payload_and_exit_code(
    tmp_path,
    thresholds_passed: bool,
    expected_exit_code: int,
    failed_thresholds: list[str],
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    config = Config(
        db_path=tmp_path / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=tmp_path,
        embedding_dims=4,
    )
    fake_store = object()
    fake_embedder = object()
    benchmark_result = BenchmarkRunResult(
        total=2,
        avg_relevance=0.9,
        avg_adherence=0.95,
        baseline_avg_relevance=0.88,
        baseline_avg_adherence=0.93,
        false_positive_rate=0.0,
        execution_success_rate=1.0,
        promotion_precision=1.0,
        thresholds_passed=thresholds_passed,
        failed_thresholds=failed_thresholds,
    )
    captured: dict[str, object] = {}

    def _fake_make_embedder(cfg: Config) -> object:
        captured["config"] = cfg
        return fake_embedder

    with (
        patch("muscle_memory.cli._load_config", return_value=config),
        patch("muscle_memory.cli._open_store", return_value=fake_store),
        patch("muscle_memory.cli._current_repo_head", return_value="abc123"),
        patch("muscle_memory.cli._current_worktree_state", return_value=(True, "clean-state")),
        patch("muscle_memory.cli._file_sha256", return_value="bench-sha"),
        patch("muscle_memory.cli.make_embedder", side_effect=_fake_make_embedder),
        patch("muscle_memory.eval.benchmark.run_benchmark", return_value=benchmark_result) as mock_run,
    ):
        result = runner.invoke(
            app,
            ["eval", "run", "--benchmark", str(benchmark_path), "--json"],
        )

    assert result.exit_code == expected_exit_code
    assert json.loads(result.output) == {
        "total": 2,
        "avg_relevance": 0.9,
        "avg_adherence": 0.95,
        "baseline_avg_relevance": 0.88,
        "baseline_avg_adherence": 0.93,
        "false_positive_rate": 0.0,
        "execution_success_rate": 1.0,
        "promotion_precision": 1.0,
        "thresholds_passed": thresholds_passed,
        "failed_thresholds": failed_thresholds,
        "improved": [],
        "degraded": [],
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": "bench-sha",
        "repo_root": str(tmp_path.resolve()),
        "repo_head": "abc123",
        "worktree_clean": True,
        "worktree_state": "clean-state",
    }
    assert captured["config"] is config
    mock_run.assert_called_once_with(fake_store, benchmark_path, embedder=fake_embedder)


def test_eval_run_exits_nonzero_when_benchmark_is_missing(tmp_path) -> None:
    config = Config(
        db_path=tmp_path / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=tmp_path,
        embedding_dims=4,
    )
    missing_path = tmp_path / "missing-benchmark.json"

    with (
        patch("muscle_memory.cli._load_config", return_value=config),
        patch("muscle_memory.cli._open_store", return_value=object()),
        patch("muscle_memory.eval.benchmark.run_benchmark") as mock_run,
    ):
        result = runner.invoke(
            app,
            ["eval", "run", "--benchmark", str(missing_path)],
        )

    assert result.exit_code == 1
    normalized = " ".join(result.output.split())
    assert "No benchmark at" in normalized
    assert missing_path.name in normalized
    assert "Run mm eval build first." in normalized
    mock_run.assert_not_called()


def test_eval_build_uses_embedder_for_benchmark_generation(tmp_path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    config = Config(
        db_path=tmp_path / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=tmp_path,
        embedding_dims=4,
    )
    fake_store = object()
    fake_embedder = object()
    captured: dict[str, object] = {}

    def _fake_make_embedder(cfg: Config) -> object:
        captured["config"] = cfg
        return fake_embedder

    with (
        patch("muscle_memory.cli._load_config", return_value=config),
        patch("muscle_memory.cli._open_store", return_value=fake_store),
        patch("muscle_memory.cli.make_embedder", side_effect=_fake_make_embedder),
        patch(
            "muscle_memory.eval.benchmark.build_benchmark",
            return_value=([], benchmark_path),
        ) as mock_build,
    ):
        result = runner.invoke(
            app,
            ["eval", "build", "--output", str(benchmark_path)],
        )

    assert result.exit_code == 0
    assert captured["config"] is config
    mock_build.assert_called_once_with(fake_store, embedder=fake_embedder, output_path=benchmark_path)
