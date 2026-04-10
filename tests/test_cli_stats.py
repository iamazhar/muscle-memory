"""Tests for the improved `mm stats` command."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from muscle_memory.cli import _relative_time, app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Episode, Maturity, Outcome, Scope, Skill, Trajectory

runner = CliRunner()


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    """Create a .claude dir with an mm.db inside, mimicking a real project."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    return tmp_path


def _make_config(store_dir: Path) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
    )


def _make_skill(
    *,
    score: float = 0.0,
    invocations: int = 0,
    successes: int = 0,
    failures: int = 0,
    maturity: Maturity = Maturity.CANDIDATE,
    activation: str = "When testing",
    refinement_count: int = 0,
    created_at: datetime | None = None,
    last_used_at: datetime | None = None,
) -> Skill:
    return Skill(
        activation=activation,
        execution="1. Do something",
        termination="Done",
        score=score,
        invocations=invocations,
        successes=successes,
        failures=failures,
        maturity=maturity,
        refinement_count=refinement_count,
        created_at=created_at or datetime.now(UTC),
        last_used_at=last_used_at,
    )


def _make_episode(
    *,
    outcome: Outcome = Outcome.SUCCESS,
    reward: float = 0.5,
    activated_skills: list[str] | None = None,
) -> Episode:
    return Episode(
        user_prompt="do something",
        trajectory=Trajectory(user_prompt="do something"),
        outcome=outcome,
        reward=reward,
        activated_skills=activated_skills or [],
    )


class TestRelativeTime:
    def test_just_now(self) -> None:
        assert _relative_time(datetime.now(UTC)) == "just now"

    def test_minutes(self) -> None:
        dt = datetime.now(UTC) - timedelta(minutes=5)
        assert _relative_time(dt) == "5m ago"

    def test_hours(self) -> None:
        dt = datetime.now(UTC) - timedelta(hours=3)
        assert _relative_time(dt) == "3h ago"

    def test_days(self) -> None:
        dt = datetime.now(UTC) - timedelta(days=10)
        assert _relative_time(dt) == "10d ago"

    def test_months(self) -> None:
        dt = datetime.now(UTC) - timedelta(days=65)
        assert _relative_time(dt) == "2mo ago"


class TestStatsEmpty:
    """Stats on an empty store should show header and empty message."""

    def test_empty_store(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "muscle-memory" in result.output
        assert "No skills yet" in result.output

    def test_empty_store_json(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["pool_used"] == 0
        assert data["episodes_total"] == 0


class TestStatsPopulated:
    """Stats with a populated store should show all sections."""

    @pytest.fixture
    def populated_store(self, store_dir: Path) -> tuple[Config, Store]:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        # Add a proven skill
        proven = _make_skill(
            activation="When setting up Python projects",
            score=0.9,
            invocations=20,
            successes=18,
            failures=2,
            maturity=Maturity.PROVEN,
            created_at=datetime.now(UTC) - timedelta(days=30),
            last_used_at=datetime.now(UTC) - timedelta(hours=2),
        )
        store.add_skill(proven)

        # Add an established skill
        established = _make_skill(
            activation="When debugging import errors",
            score=0.75,
            invocations=8,
            successes=6,
            failures=2,
            maturity=Maturity.ESTABLISHED,
            created_at=datetime.now(UTC) - timedelta(days=3),
            last_used_at=datetime.now(UTC) - timedelta(days=1),
        )
        store.add_skill(established)

        # Add a struggling skill (score < 0.5, invocations >= 3)
        struggling = _make_skill(
            activation="When running database migrations",
            score=0.2,
            invocations=5,
            successes=1,
            failures=4,
            maturity=Maturity.CANDIDATE,
            created_at=datetime.now(UTC) - timedelta(days=10),
            last_used_at=datetime.now(UTC) - timedelta(days=5),
        )
        store.add_skill(struggling)

        # Add a brand new skill (created in last 7 days)
        new_skill = _make_skill(
            activation="When adding API endpoints",
            created_at=datetime.now(UTC) - timedelta(days=1),
        )
        store.add_skill(new_skill)

        # Add episodes
        for _ in range(5):
            store.add_episode(
                _make_episode(
                    outcome=Outcome.SUCCESS,
                    reward=0.7,
                    activated_skills=[proven.id],
                )
            )
        for _ in range(3):
            store.add_episode(
                _make_episode(
                    outcome=Outcome.FAILURE,
                    reward=-0.5,
                    activated_skills=[struggling.id],
                )
            )
        for _ in range(2):
            store.add_episode(
                _make_episode(outcome=Outcome.UNKNOWN, reward=0.0)
            )

        return cfg, store

    def test_rich_output_sections(
        self, populated_store: tuple[Config, Store]
    ) -> None:
        cfg, store = populated_store

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        output = result.output

        # Header
        assert "muscle-memory" in output
        assert "active" in output
        assert "/500" in output

        # Value section
        assert "Value" in output
        assert "episodes" in output
        assert "with skills" in output

        # Learning section
        assert "Learning" in output
        assert "proven" in output
        assert "established" in output
        assert "candidate" in output
        assert "new (7d)" in output

        # Attention section
        assert "Attention" in output

        # Top Skills section
        assert "Top Skills" in output

    def test_json_output(
        self, populated_store: tuple[Config, Store]
    ) -> None:
        cfg, store = populated_store

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Structure checks
        assert data["status"] == "active"
        assert data["pool_used"] == 4
        assert data["episodes_total"] == 10
        assert data["episodes_with_skills"] == 8
        assert isinstance(data["reuse_rate"], float)
        assert isinstance(data["avg_reward"], float)
        assert "maturity" in data
        assert data["maturity"]["proven"] == 1
        assert data["maturity"]["established"] == 1
        assert data["maturity"]["candidate"] == 2
        assert data["new_7d"] >= 1
        assert "attention" in data
        assert "top_skills" in data
        assert "struggling_skills" in data

    def test_struggling_skills_shown(
        self, populated_store: tuple[Config, Store]
    ) -> None:
        cfg, store = populated_store

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert "Struggling Skills" in result.output
        assert "database migrations" in result.output


class TestStatsAttention:
    """Test the attention section flags issues correctly."""

    def test_no_issues(self, store_dir: Path) -> None:
        """A healthy store should show no attention items."""
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        # Add one healthy proven skill
        skill = _make_skill(
            score=0.9,
            invocations=15,
            successes=14,
            failures=1,
            maturity=Maturity.PROVEN,
            last_used_at=datetime.now(UTC) - timedelta(hours=1),
        )
        store.add_skill(skill)

        # All success episodes
        for _ in range(5):
            store.add_episode(
                _make_episode(outcome=Outcome.SUCCESS, activated_skills=[skill.id])
            )

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert "No issues detected" in result.output

    def test_at_risk_flagged(self, store_dir: Path) -> None:
        """Skills that will be pruned should be flagged."""
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        bad_skill = _make_skill(
            score=0.1,
            invocations=10,
            successes=1,
            failures=9,
            last_used_at=datetime.now(UTC),
        )
        store.add_skill(bad_skill)
        store.add_episode(_make_episode(outcome=Outcome.SUCCESS))

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert "at risk" in result.output

    def test_stale_flagged(self, store_dir: Path) -> None:
        """Skills unused for >30 days should be flagged."""
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        stale_skill = _make_skill(
            invocations=3,
            successes=2,
            score=0.67,
            maturity=Maturity.CANDIDATE,
            last_used_at=datetime.now(UTC) - timedelta(days=45),
        )
        store.add_skill(stale_skill)
        store.add_episode(_make_episode(outcome=Outcome.SUCCESS))

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert "stale" in result.output


class TestStatsPaused:
    """Stats should show PAUSED status when mm.paused exists."""

    def test_paused_status(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)
        store.add_skill(_make_skill())

        # Create the pause flag
        (store_dir / ".claude" / "mm.paused").touch()

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "PAUSED" in result.output

    def test_paused_json(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)
        store.add_skill(_make_skill())

        (store_dir / ".claude" / "mm.paused").touch()

        with patch("muscle_memory.cli._load_config", return_value=cfg), patch(
            "muscle_memory.cli._open_store", return_value=store
        ):
            result = runner.invoke(app, ["stats", "--json"])

        data = json.loads(result.output)
        assert data["status"] == "paused"
