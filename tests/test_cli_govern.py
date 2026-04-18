from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Scope, Skill

runner = CliRunner()


def _make_config(store_dir: Path) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
    )


def test_govern_dry_run_reports_actions(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    skill = Skill(
        id="skill_bad_health",
        activation="When tests fail",
        execution="1. Run `pytest`\n2. Fix failures",
        termination="Tests pass",
        maturity="live",
        invocations=5,
        successes=4,
        failures=1,
        score=0.8,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill)
    store.add_episode(
        __import__(
            "muscle_memory.models", fromlist=["Episode", "Outcome", "ToolCall", "Trajectory"]
        ).Episode(
            user_prompt="test prompt",
            trajectory=__import__(
                "muscle_memory.models", fromlist=["Trajectory", "ToolCall"]
            ).Trajectory(
                tool_calls=[
                    __import__("muscle_memory.models", fromlist=["ToolCall"]).ToolCall(
                        name="Bash", arguments={"command": "ls"}, result="ok"
                    )
                ]
            ),
            outcome=__import__("muscle_memory.models", fromlist=["Outcome"]).Outcome.FAILURE,
            activated_skills=[skill.id],
        )
    )
    store.add_episode(
        __import__(
            "muscle_memory.models", fromlist=["Episode", "Outcome", "ToolCall", "Trajectory"]
        ).Episode(
            user_prompt="test prompt",
            trajectory=__import__(
                "muscle_memory.models", fromlist=["Trajectory", "ToolCall"]
            ).Trajectory(
                tool_calls=[
                    __import__("muscle_memory.models", fromlist=["ToolCall"]).ToolCall(
                        name="Bash", arguments={"command": "pwd"}, result="ok"
                    )
                ]
            ),
            outcome=__import__("muscle_memory.models", fromlist=["Outcome"]).Outcome.FAILURE,
            activated_skills=[skill.id],
        )
    )
    store.add_episode(
        __import__(
            "muscle_memory.models", fromlist=["Episode", "Outcome", "ToolCall", "Trajectory"]
        ).Episode(
            user_prompt="test prompt",
            trajectory=__import__(
                "muscle_memory.models", fromlist=["Trajectory", "ToolCall"]
            ).Trajectory(
                tool_calls=[
                    __import__("muscle_memory.models", fromlist=["ToolCall"]).ToolCall(
                        name="Bash", arguments={"command": "echo ok"}, result="ok"
                    )
                ]
            ),
            outcome=__import__("muscle_memory.models", fromlist=["Outcome"]).Outcome.FAILURE,
            activated_skills=[skill.id],
        )
    )

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["maint", "govern", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["demote"] == [skill.id]
    assert data["review"] == [skill.id]


def test_govern_apply_demotes_skill(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    skill = Skill(
        id="skill_bad_health",
        activation="When tests fail",
        execution="1. Run `pytest`\n2. Fix failures",
        termination="Tests pass",
        maturity="live",
        invocations=5,
        successes=4,
        failures=1,
        score=0.8,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill)
    from muscle_memory.models import Episode, Outcome, ToolCall, Trajectory

    for cmd in ["ls", "pwd", "echo ok"]:
        store.add_episode(
            Episode(
                user_prompt="test prompt",
                trajectory=Trajectory(
                    tool_calls=[ToolCall(name="Bash", arguments={"command": cmd}, result="ok")]
                ),
                outcome=Outcome.FAILURE,
                activated_skills=[skill.id],
            )
        )

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["maint", "govern", "--apply"])

    assert result.exit_code == 0
    updated = store.get_skill(skill.id)
    assert updated is not None
    assert updated.maturity.value == "candidate"
