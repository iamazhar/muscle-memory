"""Tests for the `mm rescore` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall, Trajectory

runner = CliRunner()


def _make_config(store_dir: Path) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
    )


def test_rescore_rebuilds_invocations_from_episode_activations(tmp_path: Path) -> None:
    store_dir = tmp_path
    claude_dir = store_dir / ".claude"
    claude_dir.mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    skill = Skill(
        activation="When pytest fails with import errors",
        execution="use the test runner",
        termination="tests pass",
        invocations=1,
        successes=12,
        failures=0,
        score=1.0,
    )
    store.add_skill(skill)

    store.add_episode(
        Episode(
            user_prompt="fix tests",
            trajectory=Trajectory(
                user_prompt="fix tests",
                tool_calls=[
                    ToolCall(
                        name="Bash",
                        arguments={"command": "pytest"},
                        result="5 passed in 0.12s",
                    )
                ],
            ),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id, skill.id],
        )
    )
    store.add_episode(
        Episode(
            user_prompt="fix tests again",
            trajectory=Trajectory(
                user_prompt="fix tests again",
                tool_calls=[
                    ToolCall(
                        name="Bash",
                        arguments={"command": "pytest"},
                        error="FAILED tests/test_app.py::test_example",
                    )
                ],
            ),
            outcome=Outcome.FAILURE,
            activated_skills=[skill.id],
        )
    )

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["rescore"])

    assert result.exit_code == 0
    repaired = store.get_skill(skill.id)
    assert repaired is not None
    assert repaired.invocations == 2
    assert repaired.successes == 1
    assert repaired.failures == 1
    assert repaired.score == 0.5


def test_rescore_replays_full_episode_history(tmp_path: Path) -> None:
    store_dir = tmp_path
    claude_dir = store_dir / ".claude"
    claude_dir.mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    skill = Skill(
        activation="When pytest fails with import errors",
        execution="use the test runner",
        termination="tests pass",
    )
    store.add_skill(skill)
    store.add_episode(
        Episode(
            user_prompt="fix tests",
            trajectory=Trajectory(
                user_prompt="fix tests",
                tool_calls=[
                    ToolCall(
                        name="Bash",
                        arguments={"command": "pytest"},
                        result="5 passed in 0.12s",
                    )
                ],
            ),
            outcome=Outcome.SUCCESS,
            activated_skills=[skill.id],
        )
    )

    seen_limit: int | None = 50
    original = store.list_episodes

    def spy_list_episodes(*, limit: int | None = 50) -> list[Episode]:
        nonlocal seen_limit
        seen_limit = limit
        return original(limit=limit)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch.object(store, "list_episodes", side_effect=spy_list_episodes),
    ):
        result = runner.invoke(app, ["rescore"])

    assert result.exit_code == 0
    assert seen_limit is None
