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
]

_SUCCESS_SUBSTRINGS = [
    "passed",
    "PASS",
    "OK",
    "ok",
    "build successful",
    "tests passed",
    "All checks passed",
    "Build succeeded",
]


@dataclass
class OutcomeSignal:
    outcome: Outcome
    reward: float
    reasons: list[str]


def infer_outcome(trajectory: Trajectory, *, user_followup: str = "") -> OutcomeSignal:
    """Decide success / failure / unknown from a trajectory.

    Guiding principle: **the final state dominates**. Mid-session errors
    that were recovered from are a signal that the agent *figured something
    out* — if the last command succeeded, the episode was successful.

    Hierarchy (highest signal first):
      1. User follow-up phrasing, if present.
      2. Final tool call's error/success state + keywords in its output.
      3. Density of unrecovered tail errors.
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

    # 3. Unrecovered tail errors (2+ of the last 3 calls errored).
    last_3_errors = sum(1 for tc in tool_calls[-3:] if tc.is_error())
    if last_3_errors >= 2:
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


def _is_test_or_build(tc: ToolCall) -> bool:
    if tc.name.lower() not in {"bash", "shell", "run"}:
        return False
    cmd = str(tc.arguments.get("command", "")).lower()
    for token in ("pytest", "npm test", "npm run build", "cargo test",
                  "cargo build", "go test", "mvn test", "gradle test",
                  "yarn test", "yarn build", "make test", "make build",
                  "jest", "vitest", "tox"):
        if token in cmd:
            return True
    return False
