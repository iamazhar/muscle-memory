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


def test_python_init_path_counts_as_success() -> None:
    """Trajectory ending in a python __init__.py path should be SUCCESS —
    that's what 'import worked' looks like on disk."""
    traj = Trajectory(
        tool_calls=[
            ToolCall(name="Bash", arguments={"command": "chflags nohidden ..."}, result=""),
            ToolCall(
                name="Bash",
                arguments={"command": ".venv/bin/python3 -c 'import muscle_memory'"},
                result="/Users/x/project/src/muscle_memory/__init__.py",
            ),
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.SUCCESS


def test_modulenotfounderror_in_final_is_failure() -> None:
    """ModuleNotFoundError is a concrete failure signal even without the
    tool call being marked as an error."""
    traj = Trajectory(
        tool_calls=[
            ToolCall(
                name="Bash",
                arguments={"command": "python -c 'import foo'"},
                result="ModuleNotFoundError: No module named 'foo'",
            )
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.FAILURE


def test_skill_activated_session_defaults_to_success() -> None:
    """If a playbook was injected, the user invoked an intent. Absent
    error signals, credit the run as success. Without this, sessions
    with chain-of-Bash-success land in UNKNOWN."""
    traj = Trajectory(
        tool_calls=[
            ToolCall(name="Bash", arguments={"command": "ls -lO"}, result="-rw-r--r-- 1 u s - 49 Apr .pth"),
            ToolCall(name="Bash", arguments={"command": "chflags nohidden *.pth"}, result=""),
            ToolCall(
                name="Bash",
                arguments={"command": "python3 -c 'import x; print(x)'"},
                result="/path/to/x.py",
            ),
        ]
    )
    # without the flag, this would be UNKNOWN (no keyword hits)
    sig_no_skills = infer_outcome(traj, any_skills_activated=False)
    assert sig_no_skills.outcome is not Outcome.FAILURE
    # with the flag, it's SUCCESS
    sig_with_skills = infer_outcome(traj, any_skills_activated=True)
    assert sig_with_skills.outcome is Outcome.SUCCESS


def test_skill_activated_does_not_override_clear_failure() -> None:
    """Even with a skill activated, an errored final tool call is FAILURE."""
    traj = Trajectory(
        tool_calls=[
            ToolCall(name="Bash", arguments={"command": "x"}, error="boom"),
            ToolCall(name="Bash", arguments={"command": "y"}, error="still broken"),
        ]
    )
    sig = infer_outcome(traj, any_skills_activated=True)
    assert sig.outcome is Outcome.FAILURE
