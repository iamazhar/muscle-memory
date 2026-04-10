"""Tests for the improved `mm stats` command."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from muscle_memory.cli import _ensure_utc, _relative_time, app
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

    def test_naive_datetime(self) -> None:
        """Naive datetimes (from mm import) should not crash."""
        naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=3)
        assert _relative_time(naive) == "3d ago"


class TestEnsureUtc:
    def test_naive_gets_utc(self) -> None:
        naive = datetime(2025, 1, 1, 12, 0, 0)
        result = _ensure_utc(naive)
        assert result.tzinfo is UTC

    def test_aware_unchanged(self) -> None:
        aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = _ensure_utc(aware)
        assert result is aware


class TestStatsEmpty:
    """Stats on an empty store should show header and empty message."""

    def test_empty_store(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "muscle-memory" in result.output
        assert "No skills yet" in result.output

    def test_empty_store_json(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
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
            store.add_episode(_make_episode(outcome=Outcome.UNKNOWN, reward=0.0))

        return cfg, store

    def test_rich_output_sections(self, populated_store: tuple[Config, Store]) -> None:
        cfg, store = populated_store

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
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

    def test_json_output(self, populated_store: tuple[Config, Store]) -> None:
        cfg, store = populated_store

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
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

    def test_struggling_skills_shown(self, populated_store: tuple[Config, Store]) -> None:
        cfg, store = populated_store

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
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
            store.add_episode(_make_episode(outcome=Outcome.SUCCESS, activated_skills=[skill.id]))

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
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

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
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

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats"])

        assert "stale" in result.output


class TestStatsNaiveDatetimes:
    """Skills imported via mm import may have naive (offset-unaware) timestamps."""

    def test_naive_created_at_does_not_crash(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)

        # Simulate a skill with naive datetime (as mm import can produce)
        naive_skill = _make_skill(
            activation="When doing something",
            invocations=3,
            successes=2,
            score=0.67,
            created_at=datetime(2025, 6, 15, 10, 0, 0),  # no tzinfo
            last_used_at=datetime(2025, 6, 20, 10, 0, 0),  # no tzinfo
        )
        store.add_skill(naive_skill)
        store.add_episode(_make_episode(outcome=Outcome.SUCCESS))

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Learning" in result.output


class TestStatsPaused:
    """Stats should show PAUSED status when mm.paused exists."""

    def test_paused_status(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)
        store.add_skill(_make_skill())

        # Create the pause flag
        (store_dir / ".claude" / "mm.paused").touch()

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "PAUSED" in result.output

    def test_paused_json(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)
        store.add_skill(_make_skill())

        (store_dir / ".claude" / "mm.paused").touch()

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats", "--json"])

        data = json.loads(result.output)
        assert data["status"] == "paused"


# ------------------------------------------------------------------
# Core metric computation tests (via --json for exact value checks)
# ------------------------------------------------------------------


def _json_stats(store_dir: Path, skills: list[Skill], episodes: list[Episode]) -> dict:
    """Helper: seed a store, run stats --json, return parsed dict."""
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)
    for s in skills:
        store.add_skill(s)
    for ep in episodes:
        store.add_episode(ep)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["stats", "--json"])
    assert result.exit_code == 0
    return json.loads(result.output)


class TestReuseRate:
    def test_basic_rate(self, store_dir: Path) -> None:
        s = _make_skill(invocations=10, successes=7, failures=3, score=0.7)
        data = _json_stats(store_dir, [s], [_make_episode()])
        assert data["reuse_rate"] == pytest.approx(0.7)

    def test_zero_invocations(self, store_dir: Path) -> None:
        s = _make_skill(invocations=0, successes=0)
        data = _json_stats(store_dir, [s], [_make_episode()])
        assert data["reuse_rate"] == 0.0

    def test_aggregates_across_skills(self, store_dir: Path) -> None:
        """Reuse rate sums across all skills, not per-skill average."""
        s1 = _make_skill(activation="When A", invocations=10, successes=8, score=0.8)
        s2 = _make_skill(activation="When B", invocations=10, successes=2, score=0.2)
        data = _json_stats(store_dir, [s1, s2], [_make_episode()])
        # 10 total successes / 20 total invocations = 0.5
        assert data["reuse_rate"] == pytest.approx(0.5)


class TestAvgReward:
    def test_only_skill_activated_episodes(self, store_dir: Path) -> None:
        """Avg reward only considers episodes where skills were activated."""
        s = _make_skill(invocations=1, successes=1, score=1.0)
        eps = [
            _make_episode(reward=0.8, activated_skills=[s.id]),
            _make_episode(reward=0.6, activated_skills=[s.id]),
            _make_episode(reward=-0.5),  # unassisted — excluded
        ]
        data = _json_stats(store_dir, [s], eps)
        assert data["avg_reward"] == pytest.approx(0.7)

    def test_no_activated_episodes(self, store_dir: Path) -> None:
        s = _make_skill()
        eps = [_make_episode(reward=0.5)]  # no activated_skills
        data = _json_stats(store_dir, [s], eps)
        assert data["avg_reward"] == 0.0

    def test_negative_rewards_pull_average_down(self, store_dir: Path) -> None:
        s = _make_skill(invocations=3, successes=1, failures=2, score=0.33)
        eps = [
            _make_episode(reward=0.8, activated_skills=[s.id]),
            _make_episode(reward=-0.6, activated_skills=[s.id]),
            _make_episode(reward=-0.4, activated_skills=[s.id]),
        ]
        data = _json_stats(store_dir, [s], eps)
        assert data["avg_reward"] == pytest.approx(-0.2 / 3)


class TestEpisodesWithSkills:
    def test_counts_and_percentage(self, store_dir: Path) -> None:
        s = _make_skill(invocations=2, successes=2, score=1.0)
        eps = [
            _make_episode(activated_skills=[s.id]),
            _make_episode(activated_skills=[s.id]),
            _make_episode(),  # unassisted
            _make_episode(),  # unassisted
            _make_episode(),  # unassisted
        ]
        data = _json_stats(store_dir, [s], eps)
        assert data["episodes_with_skills"] == 2
        assert data["episodes_with_skills_pct"] == pytest.approx(0.4)

    def test_no_episodes(self, store_dir: Path) -> None:
        s = _make_skill()
        data = _json_stats(store_dir, [s], [])
        assert data["episodes_with_skills"] == 0
        assert data["episodes_with_skills_pct"] == 0.0


class TestEpisodesTotalCount:
    def test_true_count_not_capped(self, store_dir: Path) -> None:
        """episodes_total should reflect Store.count_episodes(), not len(windowed)."""
        s = _make_skill()
        # We can't easily exceed 1000 in a unit test, but we can verify the
        # field comes from count_episodes() by checking it matches the number
        # we inserted.
        eps = [_make_episode() for _ in range(7)]
        data = _json_stats(store_dir, [s], eps)
        assert data["episodes_total"] == 7


class TestNew7d:
    def test_counts_recent_skills(self, store_dir: Path) -> None:
        now = datetime.now(UTC)
        skills = [
            _make_skill(activation="When A", created_at=now - timedelta(days=1)),
            _make_skill(activation="When B", created_at=now - timedelta(days=6)),
            _make_skill(activation="When C", created_at=now - timedelta(days=8)),  # outside 7d
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["new_7d"] == 2

    def test_boundary_just_inside_7_days(self, store_dir: Path) -> None:
        now = datetime.now(UTC)
        # Just inside the window (6d 23h ago) — should count
        skills = [
            _make_skill(
                activation="When X",
                created_at=now - timedelta(days=6, hours=23),
            ),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["new_7d"] == 1

    def test_all_old(self, store_dir: Path) -> None:
        skills = [
            _make_skill(created_at=datetime.now(UTC) - timedelta(days=30)),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["new_7d"] == 0


class TestRefinedMetrics:
    def test_counts_refined_skills_and_total(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation="When A", refinement_count=3),
            _make_skill(activation="When B", refinement_count=1),
            _make_skill(activation="When C", refinement_count=0),  # not refined
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["refined_skills"] == 2
        assert data["total_refinements"] == 4


class TestAttentionMetrics:
    def test_need_refine_matches_criteria(self, store_dir: Path) -> None:
        """need_refine: invocations >= 5, failures >= 2, score <= 0.6."""
        skills = [
            # meets all criteria
            _make_skill(
                activation="When A",
                invocations=5,
                successes=2,
                failures=3,
                score=0.4,
            ),
            # score too high
            _make_skill(
                activation="When B",
                invocations=5,
                successes=4,
                failures=2,
                score=0.8,
            ),
            # not enough invocations
            _make_skill(
                activation="When C",
                invocations=3,
                successes=1,
                failures=2,
                score=0.33,
            ),
            # not enough failures
            _make_skill(
                activation="When D",
                invocations=6,
                successes=3,
                failures=1,
                score=0.5,
            ),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["attention"]["need_refine"] == 1

    def test_at_risk_boundary(self, store_dir: Path) -> None:
        """at_risk: invocations >= 5 and score <= 0.2."""
        skills = [
            _make_skill(activation="When A", invocations=5, score=0.2),  # exactly at boundary
            _make_skill(activation="When B", invocations=5, score=0.21),  # just above
            _make_skill(activation="When C", invocations=4, score=0.1),  # not enough invocations
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["attention"]["at_risk"] == 1

    def test_stale_requires_invocations_and_old_use(self, store_dir: Path) -> None:
        """Stale: invocations > 0, last_used_at not None, and > 30d ago."""
        now = datetime.now(UTC)
        skills = [
            # stale: used but not in 30+ days
            _make_skill(
                activation="When A",
                invocations=3,
                last_used_at=now - timedelta(days=35),
            ),
            # not stale: never invoked
            _make_skill(
                activation="When B",
                invocations=0,
                last_used_at=None,
            ),
            # not stale: used recently
            _make_skill(
                activation="When C",
                invocations=5,
                last_used_at=now - timedelta(days=5),
            ),
            # not stale: invoked but last_used_at is None (edge case)
            _make_skill(
                activation="When D",
                invocations=2,
                last_used_at=None,
            ),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["attention"]["stale"] == 1

    def test_unknown_rate(self, store_dir: Path) -> None:
        s = _make_skill()
        eps = [
            _make_episode(outcome=Outcome.UNKNOWN),
            _make_episode(outcome=Outcome.UNKNOWN),
            _make_episode(outcome=Outcome.UNKNOWN),
            _make_episode(outcome=Outcome.SUCCESS),
            _make_episode(outcome=Outcome.FAILURE),
        ]
        data = _json_stats(store_dir, [s], eps)
        assert data["attention"]["unknown_rate"] == pytest.approx(0.6)

    def test_unknown_rate_shown_in_rich_only_above_40pct(self, store_dir: Path) -> None:
        """The unknown rate line only appears in Rich output when > 40%."""
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)
        store.add_skill(_make_skill())
        # 30% unknown — below threshold
        for _ in range(7):
            store.add_episode(_make_episode(outcome=Outcome.SUCCESS))
        for _ in range(3):
            store.add_episode(_make_episode(outcome=Outcome.UNKNOWN))

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats"])

        assert "unknown rate" not in result.output


class TestTopSkillsSelection:
    def test_requires_min_2_invocations(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation="When A", invocations=1, successes=1, score=1.0),
            _make_skill(activation="When B", invocations=2, successes=2, score=1.0),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert len(data["top_skills"]) == 1
        assert data["top_skills"][0]["activation"] == "When B"

    def test_limited_to_3(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation=f"When {c}", invocations=5, successes=4, score=0.8)
            for c in "ABCDE"
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert len(data["top_skills"]) == 3

    def test_sorted_by_score_desc(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation="When Low", invocations=5, successes=2, score=0.4),
            _make_skill(activation="When High", invocations=5, successes=5, score=1.0),
            _make_skill(activation="When Mid", invocations=5, successes=3, score=0.6),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        scores = [s["score"] for s in data["top_skills"]]
        assert scores == sorted(scores, reverse=True)


class TestStrugglingSkillsSelection:
    def test_requires_min_3_invocations_and_low_score(self, store_dir: Path) -> None:
        skills = [
            # qualifies: invocations >= 3, score < 0.5
            _make_skill(activation="When Bad", invocations=4, successes=1, score=0.25),
            # too few invocations
            _make_skill(activation="When New", invocations=2, successes=0, score=0.0),
            # score too high
            _make_skill(activation="When OK", invocations=5, successes=3, score=0.6),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert len(data["struggling_skills"]) == 1
        assert data["struggling_skills"][0]["activation"] == "When Bad"

    def test_limited_to_3(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation=f"When {c}", invocations=5, successes=1, score=0.2)
            for c in "ABCDE"
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert len(data["struggling_skills"]) == 3

    def test_sorted_by_score_asc(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation="When Mid", invocations=5, successes=2, score=0.4),
            _make_skill(activation="When Worst", invocations=5, successes=0, score=0.0),
            _make_skill(activation="When Bad", invocations=5, successes=1, score=0.2),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        scores = [s["score"] for s in data["struggling_skills"]]
        assert scores == sorted(scores)

    def test_empty_when_all_healthy(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation="When Good", invocations=10, successes=8, score=0.8),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["struggling_skills"] == []

    def test_section_omitted_in_rich_when_none(self, store_dir: Path) -> None:
        cfg = _make_config(store_dir)
        store = Store(cfg.db_path)
        store.add_skill(
            _make_skill(invocations=10, successes=9, score=0.9, maturity=Maturity.PROVEN)
        )
        store.add_episode(_make_episode())

        with (
            patch("muscle_memory.cli._load_config", return_value=cfg),
            patch("muscle_memory.cli._open_store", return_value=store),
        ):
            result = runner.invoke(app, ["stats"])

        assert "Struggling Skills" not in result.output


class TestMaturityBreakdown:
    def test_counts_per_level(self, store_dir: Path) -> None:
        skills = [
            _make_skill(activation="When A", maturity=Maturity.PROVEN),
            _make_skill(activation="When B", maturity=Maturity.PROVEN),
            _make_skill(activation="When C", maturity=Maturity.ESTABLISHED),
            _make_skill(activation="When D", maturity=Maturity.CANDIDATE),
            _make_skill(activation="When E", maturity=Maturity.CANDIDATE),
            _make_skill(activation="When F", maturity=Maturity.CANDIDATE),
        ]
        data = _json_stats(store_dir, skills, [_make_episode()])
        assert data["maturity"] == {
            "proven": 2,
            "established": 1,
            "candidate": 3,
        }
