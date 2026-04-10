"""Tests for the outcome heuristic."""

from __future__ import annotations

from muscle_memory.models import Outcome, ToolCall, Trajectory
from muscle_memory.outcomes import _detect_git_success, _parse_test_results, infer_outcome


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
            ToolCall(
                name="Bash",
                arguments={"command": "ls -lO"},
                result="-rw-r--r-- 1 u s - 49 Apr .pth",
            ),
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


class TestParseTestResults:
    def test_pytest_all_pass(self) -> None:
        r = _parse_test_results("121 passed in 0.49s")
        assert r is not None
        assert r.passed == 121 and r.failed == 0 and r.runner == "pytest"

    def test_pytest_mixed(self) -> None:
        r = _parse_test_results("119 passed, 2 failed in 1.23s")
        assert r is not None
        assert r.passed == 119 and r.failed == 2

    def test_jest_pass(self) -> None:
        r = _parse_test_results("Tests: 5 passed, 5 total")
        assert r is not None
        assert r.passed == 5 and r.failed == 0 and r.runner == "jest"

    def test_jest_mixed(self) -> None:
        r = _parse_test_results("Tests: 2 failed, 8 passed, 10 total")
        assert r is not None
        assert r.passed == 8 and r.failed == 2

    def test_cargo(self) -> None:
        r = _parse_test_results("test result: ok. 10 passed; 0 failed; 0 ignored")
        assert r is not None
        assert r.passed == 10 and r.failed == 0 and r.runner == "cargo"

    def test_go_pass(self) -> None:
        r = _parse_test_results("ok  \tgithub.com/foo/bar\t0.003s\n")
        assert r is not None
        assert r.passed == 1 and r.failed == 0 and r.runner == "go"

    def test_go_fail(self) -> None:
        r = _parse_test_results("FAIL\tgithub.com/foo/bar\t0.003s\n")
        assert r is not None
        assert r.failed == 1

    def test_no_match(self) -> None:
        assert _parse_test_results("just some random output") is None

    def test_bare_passed_without_time_no_match(self) -> None:
        """'N passed' without 'in Xs' should not match pytest pattern."""
        assert _parse_test_results("5 checks passed") is None


class TestDetectGitSuccess:
    def test_commit_with_hash(self) -> None:
        calls = [
            ToolCall(
                name="Bash",
                arguments={"command": "git commit -m 'fix bug'"},
                result="[main abc1234] fix bug\n 1 file changed",
            ),
        ]
        assert _detect_git_success(calls) == 2.5

    def test_push_with_hash_range(self) -> None:
        calls = [
            ToolCall(
                name="Bash",
                arguments={"command": "git push origin main"},
                result="abc1234..def5678 main -> main",
            ),
        ]
        assert _detect_git_success(calls) == 2.5

    def test_hex_in_non_git_output_no_match(self) -> None:
        """Hex strings in non-git-format output should not match."""
        calls = [
            ToolCall(
                name="Bash",
                arguments={"command": "git commit -m 'x'"},
                result="Everything up-to-date",
            ),
        ]
        assert _detect_git_success(calls) == 0.0

    def test_commit_error_gives_no_bonus(self) -> None:
        calls = [
            ToolCall(
                name="Bash",
                arguments={"command": "git commit -m 'x'"},
                error="nothing to commit",
            ),
        ]
        assert _detect_git_success(calls) == 0.0

    def test_no_git_calls(self) -> None:
        calls = [
            ToolCall(name="Bash", arguments={"command": "ls"}, result="file.py"),
        ]
        assert _detect_git_success(calls) == 0.0


def test_structured_pytest_gives_stronger_signal() -> None:
    """Structured test parsing should give +4.0 vs keyword's +3.0."""
    traj = Trajectory(
        tool_calls=[
            ToolCall(
                name="Bash",
                arguments={"command": "pytest"},
                result="47 passed in 3.21s",
            ),
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.SUCCESS
    assert any("pytest" in r for r in sig.reasons)


def test_partial_test_results_moderate_signal() -> None:
    """Some passes + some failures = moderate positive signal (+1.0)."""
    traj = Trajectory(
        tool_calls=[
            ToolCall(
                name="Bash",
                arguments={"command": "pytest"},
                result="45 passed, 2 failed in 3.21s",
            ),
        ]
    )
    sig = infer_outcome(traj)
    # +1.0 alone is not enough for SUCCESS threshold (2.0)
    assert sig.outcome is Outcome.UNKNOWN


def test_git_commit_boosts_outcome() -> None:
    traj = Trajectory(
        tool_calls=[
            ToolCall(name="Bash", arguments={"command": "ls"}, result="file.py"),
            ToolCall(
                name="Bash",
                arguments={"command": "git commit -m 'fix'"},
                result="[main abc1234] fix\n 1 file changed",
            ),
        ]
    )
    sig = infer_outcome(traj)
    assert sig.outcome is Outcome.SUCCESS
    assert any("git" in r for r in sig.reasons)


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
