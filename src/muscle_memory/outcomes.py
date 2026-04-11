"""Heuristic outcome detection for agent episodes.

Claude Code doesn't hand us an explicit success/failure signal when
a session ends, so we infer one from the trajectory.

The heuristic is intentionally biased toward UNKNOWN. We'd rather
miss a few successes than poison the skill store with wrong scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from muscle_memory.models import Outcome, ToolCall, Trajectory

# user-correction phrases suggest the previous attempt was wrong
_CORRECTION_PATTERNS = [
    r"\bno,?\s+that'?s wrong\b",
    r"\bthat'?s not (what|right)\b",
    r"\bthat did(n'?t|not) work\b",
    r"\bstop\b",
    r"\bundo\b",
    r"\brevert\b",
    r"\byou broke\b",
    r"\bwhy did you\b",
    r"\bdon'?t do that\b",
    r"\bthat'?s not correct\b",
]

# positive feedback signals
_POSITIVE_PATTERNS = [
    r"\bperfect\b",
    r"\bthanks?\b",
    r"\blooks good\b",
    r"\bnice\b",
    r"\bgreat\b",
    r"\bthat worked\b",
    r"\bexactly\b",
    r"\bship it\b",
    r"\bmerge(d)?\b",
]

# test / build failure substrings in tool output
_FAIL_SUBSTRINGS = [
    "FAILED",
    "error:",
    "ERROR:",
    "Traceback (most recent call last)",
    "exit code: 1",
    "npm ERR!",
    "BUILD FAILED",
    "TEST FAILED",
    "AssertionError",
    "SyntaxError",
    "panic:",
    "ModuleNotFoundError",
    "ImportError",
    "command not found",
    "No such file",
    "Permission denied",
]

_SUCCESS_SUBSTRINGS = [
    "passed",
    "PASS",
    "OK",
    " ok\n",
    "build successful",
    "tests passed",
    "All checks passed",
    "Build succeeded",
    # "exit code: 0" and friends
    "exit 0",
    "exit code 0",
    "exited with code 0",
    # Python package/module resolution: seeing an __init__.py path in output
    # almost always means a successful import.
    "__init__.py",
]


@dataclass
class OutcomeSignal:
    outcome: Outcome
    reward: float
    reasons: list[str]


def infer_outcome(
    trajectory: Trajectory,
    *,
    user_followup: str = "",
    any_skills_activated: bool = False,
) -> OutcomeSignal:
    """Decide success / failure / unknown from a trajectory.

    Guiding principle: **the final state dominates**. Mid-session errors
    that were recovered from are a signal that the agent *figured something
    out* — if the last command succeeded, the episode was successful.

    Hierarchy (highest signal first):
      1. User follow-up phrasing, if present.
      2. Final tool call's error/success state + keywords in its output.
      3. Skill-activated sessions default toward success when nothing
         obviously failed — the user invoked a playbook, the playbook
         executed, no tail errors. Score it.
      4. Density of unrecovered tail errors.
    """
    reasons: list[str] = []
    score = 0.0

    tool_calls = trajectory.tool_calls
    if not tool_calls:
        return OutcomeSignal(outcome=Outcome.UNKNOWN, reward=0.0, reasons=reasons)

    # 1. User follow-up language is the clearest signal.
    if user_followup:
        low = user_followup.lower()
        if _any_regex_hit(low, _CORRECTION_PATTERNS):
            score -= 3.0
            reasons.append("user follow-up contains correction phrasing")
        elif _any_regex_hit(low, _POSITIVE_PATTERNS):
            score += 3.0
            reasons.append("user follow-up contains positive phrasing")

    # 2. Final tool call state.
    last = tool_calls[-1]
    last_text = (last.result or "") + (last.error or "")

    if last.is_error():
        score -= 3.0
        reasons.append("final tool call errored")
    else:
        # Try structured test runner parsing first (stronger signal).
        test_result = _parse_test_results(last_text)
        if test_result is not None:
            if test_result.failed == 0 and test_result.passed > 0:
                score += 4.0
                reasons.append(f"{test_result.runner}: {test_result.passed} passed, 0 failed")
            elif test_result.failed > 0 and test_result.passed > 0:
                score += 1.0
                reasons.append(
                    f"{test_result.runner}: {test_result.passed} passed, "
                    f"{test_result.failed} failed (partial)"
                )
            elif test_result.failed > 0:
                score -= 3.0
                reasons.append(f"{test_result.runner}: 0 passed, {test_result.failed} failed")
        else:
            # Fallback to keyword matching.
            has_success = _contains_any(last_text, _SUCCESS_SUBSTRINGS)
            has_fail = _contains_any(last_text, _FAIL_SUBSTRINGS)
            if has_success and not has_fail:
                score += 3.0
                reasons.append("success keywords in final tool output")
            elif has_fail:
                score -= 2.0
                reasons.append("failure keywords in final tool output")
            elif last_text.strip():
                # finished cleanly with some output, no directional keywords
                score += 0.5
                reasons.append("final tool call completed without error")

    # 2b. Git commit/push success in recent calls.
    # Only apply if stage 2 did not already produce a strong positive,
    # to avoid score inflation from stacking signals.
    git_bonus = _detect_git_success(tool_calls)
    if git_bonus > 0 and score < 2.0:
        score += git_bonus
        reasons.append("successful git commit/push detected")

    # 3. Skill-activated bias: if the user invoked a playbook (a skill was
    # retrieved + injected), the expected outcome is success. Credit the
    # run when nothing obviously failed. Without this, pure-Bash success
    # chains like `chflags && .venv/bin/python3 -c 'import x'` land in
    # UNKNOWN even though they clearly worked.
    if any_skills_activated:
        last_3_errors_count = sum(1 for tc in tool_calls[-3:] if tc.is_error())
        last_text_has_fail = _contains_any(last_text, _FAIL_SUBSTRINGS)
        if (
            not last.is_error()
            and not last_text_has_fail
            and last_3_errors_count <= 1
            and last_text.strip()
        ):
            score += 1.5
            reasons.append("skill-activated session ended without tail errors (implicit success)")

    # 4. Unrecovered tail errors (2+ of the last 3 calls errored).
    # Skip this penalty if the final tool call was a clear success —
    # in a recovery trajectory, earlier errors are the point, not a
    # reason to demote.
    last_3_errors = sum(1 for tc in tool_calls[-3:] if tc.is_error())
    final_call_clearly_succeeded = (
        not last.is_error()
        and _contains_any(last_text, _SUCCESS_SUBSTRINGS)
        and not _contains_any(last_text, _FAIL_SUBSTRINGS)
    )
    if last_3_errors >= 2 and not final_call_clearly_succeeded:
        score -= 1.5
        reasons.append(f"{last_3_errors} tool errors in last 3 calls (unrecovered)")

    # Decide final outcome.
    if score >= 2.0:
        outcome = Outcome.SUCCESS
        reward = min(1.0, score / 5.0)
    elif score <= -2.0:
        outcome = Outcome.FAILURE
        reward = max(-1.0, score / 5.0)
    else:
        outcome = Outcome.UNKNOWN
        reward = 0.0

    return OutcomeSignal(outcome=outcome, reward=reward, reasons=reasons)


def _contains_any(text: str, substrings: list[str]) -> bool:
    return any(sub in text for sub in substrings)


def _any_regex_hit(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


# Structured test runner output patterns
# Order: most specific prefix first, pytest last (broadest).
_TEST_RUNNER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # cargo test: "test result: ok. 5 passed; 0 failed"
    ("cargo", re.compile(r"test result: \w+\.\s+(\d+) passed;\s+(\d+) failed")),
    # jest: "Tests: 5 passed, 2 failed, 7 total"
    ("jest", re.compile(r"Tests:\s+(?:(\d+) failed,\s+)?(\d+) passed")),
    # go test: "ok" (pass) or "FAIL" per package
    ("go", re.compile(r"^(ok|FAIL)\s+\S+", re.MULTILINE)),
    # pytest: "5 passed in 1.2s", "5 passed, 2 failed in 1.2s" — broadest, must be last
    ("pytest", re.compile(r"(\d+) passed(?:.*?(\d+) failed)?.*?in \d+")),
]


@dataclass
class TestResult:
    passed: int
    failed: int
    runner: str


def _parse_test_results(text: str) -> TestResult | None:
    """Try to extract structured pass/fail counts from test runner output."""
    for runner, pattern in _TEST_RUNNER_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if runner == "pytest":
            passed = int(m.group(1))
            failed = int(m.group(2)) if m.group(2) else 0
            return TestResult(passed=passed, failed=failed, runner=runner)
        if runner == "jest":
            # jest groups: (failed?, passed)
            failed = int(m.group(1)) if m.group(1) else 0
            passed = int(m.group(2))
            return TestResult(passed=passed, failed=failed, runner=runner)
        if runner == "cargo":
            passed = int(m.group(1))
            failed = int(m.group(2))
            return TestResult(passed=passed, failed=failed, runner=runner)
        if runner == "go":
            # Count ok/FAIL lines across all packages
            oks = len(re.findall(r"^ok\s+\S+", text, re.MULTILINE))
            fails = len(re.findall(r"^FAIL\s+\S+", text, re.MULTILINE))
            if oks or fails:
                return TestResult(passed=oks, failed=fails, runner=runner)
    return None


def _detect_git_success(tool_calls: list[ToolCall]) -> float:
    """Scan recent tool calls for successful git commit/push. Returns score bonus."""
    for tc in tool_calls[-5:]:
        if tc.name.lower() not in {"bash", "shell", "run"}:
            continue
        cmd = str(tc.arguments.get("command", "")).lower()
        if not any(k in cmd for k in ("git commit", "git push", "git merge")):
            continue
        if tc.is_error():
            continue
        result = tc.result or ""
        # Match git commit output format: [branch hash] or push format: hash..hash
        if re.search(r"\[[^\]]+\s+[0-9a-f]{7,40}\]", result) or re.search(
            r"[0-9a-f]{7,40}\.\.[0-9a-f]{7,40}", result
        ):
            return 2.5
    return 0.0


def _is_test_or_build(tc: ToolCall) -> bool:
    if tc.name.lower() not in {"bash", "shell", "run"}:
        return False
    cmd = str(tc.arguments.get("command", "")).lower()
    for token in (
        "pytest",
        "npm test",
        "npm run build",
        "cargo test",
        "cargo build",
        "go test",
        "mvn test",
        "gradle test",
        "yarn test",
        "yarn build",
        "make test",
        "make build",
        "jest",
        "vitest",
        "tox",
    ):
        if token in cmd:
            return True
    return False
