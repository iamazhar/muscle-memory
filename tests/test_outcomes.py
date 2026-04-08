"""Tests for the outcome heuristic."""

from __future__ import annotations

from muscle_memory.models import Outcome, ToolCall, Trajectory
from muscle_memory.outcomes import infer_outcome


def test_empty_trajectory_is_unknown() -> None:
    sig = infer_outcome(Trajectory())
    assert sig.outcome is Outcome.UNKNOWN


def test_pytest_failure_then_recovery_is_success(successful_trajectory: Trajectory) -> None:
    sig = infer_outcome(successful_trajectory)
    assert sig.outcome is Outcome.SUCCESS
    assert sig.reward > 0


def test_final_error_is_failure() -> None:
    traj = Trajectory(
        tool_calls=[
            ToolCall(name="Bash", arguments={"command": "pytest"}, error="crash"),
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.FAILURE
    assert sig.reward < 0


def test_two_tail_errors_is_failure() -> None:
    traj = Trajectory(
        tool_calls=[
            ToolCall(name="Read", result="ok"),
            ToolCall(name="Bash", error="nope"),
            ToolCall(name="Bash", error="still nope"),
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.FAILURE


def test_user_correction_marks_failure() -> None:
    traj = Trajectory(
        tool_calls=[ToolCall(name="Bash", arguments={"command": "ls"}, result="a.py")]
    )
    sig = infer_outcome(traj, user_followup="No, that's wrong. Undo that.")
    assert sig.outcome is Outcome.FAILURE


def test_user_positive_marks_success() -> None:
    traj = Trajectory(
        tool_calls=[ToolCall(name="Bash", arguments={"command": "ls"}, result="a.py")]
    )
    sig = infer_outcome(traj, user_followup="Perfect, thanks")
    assert sig.outcome is Outcome.SUCCESS


def test_build_success_keywords() -> None:
    traj = Trajectory(
        tool_calls=[
            ToolCall(
                name="Bash",
                arguments={"command": "npm run build"},
                result="build successful",
            ),
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.SUCCESS
