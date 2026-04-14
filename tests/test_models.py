"""Tests for muscle_memory.models."""

from __future__ import annotations

import pytest

from muscle_memory.models import (
    Episode,
    Maturity,
    Outcome,
    Skill,
    ToolCall,
    Trajectory,
)


class TestSkill:
    def test_activation_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError):
            Skill(activation="", execution="step", termination="done")
        with pytest.raises(ValueError):
            Skill(activation="   ", execution="step", termination="done")

    def test_recompute_score_no_invocations(self) -> None:
        s = Skill(activation="a", execution="e", termination="t")
        s.recompute_score()
        assert s.score == 0.0

    def test_recompute_score_mix(self) -> None:
        s = Skill(activation="a", execution="e", termination="t")
        s.invocations = 10
        s.successes = 7
        s.failures = 3
        s.recompute_score()
        assert s.score == pytest.approx(0.7)

    def test_recompute_score_clamps_to_one(self) -> None:
        s = Skill(activation="a", execution="e", termination="t")
        s.invocations = 1
        s.successes = 5
        s.recompute_score()
        assert s.score == 1.0

    def test_score_is_clamped_on_load(self) -> None:
        s = Skill(activation="a", execution="e", termination="t", score=12.0)
        assert s.score == 1.0

    def test_maturity_candidate_then_live_then_proven(self) -> None:
        s = Skill(activation="a", execution="e", termination="t")

        # candidate initially
        s.recompute_maturity()
        assert s.maturity is Maturity.CANDIDATE

        # still candidate with only one supporting source episode
        s.invocations = 2
        s.successes = 2
        s.source_episode_ids = ["ep1"]
        s.recompute_score()
        s.recompute_maturity()
        assert s.maturity is Maturity.CANDIDATE

        # live at 2 successes with score >= 0.75 and repeated evidence
        s.source_episode_ids = ["ep1", "ep2"]
        s.recompute_maturity()
        assert s.maturity is Maturity.LIVE

        # proven at 10 successes with score >= 0.7
        s.invocations = 13
        s.successes = 10
        s.source_episode_ids = [f"ep{i}" for i in range(1, 11)]
        s.recompute_score()
        s.recompute_maturity()
        assert s.maturity is Maturity.PROVEN

    def test_maturity_demotes_when_score_falls(self) -> None:
        s = Skill(
            activation="a",
            execution="e",
            termination="t",
            invocations=3,
            successes=3,
            source_episode_ids=["ep1", "ep2", "ep3"],
        )
        s.recompute_score()
        s.recompute_maturity()
        assert s.maturity is Maturity.LIVE

        # failures drag score below threshold -> candidate
        s.failures = 10
        s.invocations = 13
        s.recompute_score()
        s.recompute_maturity()
        assert s.maturity is Maturity.CANDIDATE

    def test_proven_demotes_to_live_before_candidate(self) -> None:
        s = Skill(
            activation="a",
            execution="e",
            termination="t",
            invocations=12,
            successes=10,
            failures=2,
            source_episode_ids=[f"ep{i}" for i in range(10)],
        )
        s.recompute_score()
        s.recompute_maturity()
        assert s.maturity is Maturity.PROVEN

        s.invocations = 15
        s.successes = 10
        s.failures = 5
        s.recompute_score()
        s.recompute_maturity()
        assert s.maturity is Maturity.LIVE


class TestTrajectory:
    def test_num_tool_calls(self) -> None:
        t = Trajectory(tool_calls=[ToolCall(name="Bash"), ToolCall(name="Read")])
        assert t.num_tool_calls() == 2

    def test_errored_tool_calls(self) -> None:
        t = Trajectory(
            tool_calls=[
                ToolCall(name="Bash", error="oops"),
                ToolCall(name="Read", result="ok"),
            ]
        )
        errs = t.errored_tool_calls()
        assert len(errs) == 1
        assert errs[0].name == "Bash"


def test_episode_defaults(successful_trajectory: Trajectory) -> None:
    ep = Episode(user_prompt="x", trajectory=successful_trajectory)
    assert ep.outcome is Outcome.UNKNOWN
    assert ep.reward == 0.0
    assert len(ep.id) == 12
