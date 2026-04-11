"""Admission policy for extracted skills.

This is the first line of defense against junk skills. The policy is
intentionally conservative:

  * Only clearly successful, non-trivial episodes may teach skills.
  * Extracted skills must look like reusable procedures, not one-off tasks.
  * If we're unsure, we reject the candidate. False negatives are cheaper
    than polluting the retrievable pool with junk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from muscle_memory.models import Episode, Maturity, Outcome, Skill

MIN_TOOL_CALLS_FOR_EXTRACTION = 2
MIN_EXECUTION_STEPS = 2
MIN_ACTIVATION_WORDS = 3
MIN_SOURCE_EPISODES_FOR_LIVE = 2

_ONE_OFF_LITERAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/(?:private/)?tmp/|/var/folders/", re.IGNORECASE), "temp path"),
    (re.compile(r"\bsession[_ -]?id\b", re.IGNORECASE), "session identifier"),
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        "UUID",
    ),
    (
        re.compile(r"\b(?:pr|pull request|issue)\s*#?\d+\b", re.IGNORECASE),
        "issue/PR reference",
    ),
    (re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"), "exact date"),
]


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str | None = None


def should_extract_from_episode(episode: Episode) -> AdmissionDecision:
    """Decide whether an episode is high-signal enough to teach from."""
    if episode.outcome is not Outcome.SUCCESS:
        return AdmissionDecision(False, "episode was not a clear success")
    if len(episode.trajectory.tool_calls) < MIN_TOOL_CALLS_FOR_EXTRACTION:
        return AdmissionDecision(False, "episode was too small to show a reusable procedure")
    return AdmissionDecision(True)


def admit_extracted_skill(skill: Skill) -> AdmissionDecision:
    """Reject extracted candidates that look one-off or under-specified."""
    if len(_activation_words(skill.activation)) < MIN_ACTIVATION_WORDS:
        return AdmissionDecision(False, "activation was too vague")

    if _execution_step_count(skill.execution) < MIN_EXECUTION_STEPS:
        return AdmissionDecision(False, "execution did not contain enough concrete steps")

    for text in (skill.activation, skill.execution, skill.termination):
        for pattern, label in _ONE_OFF_LITERAL_PATTERNS:
            if pattern.search(text):
                return AdmissionDecision(False, f"contains one-off {label}")

    return AdmissionDecision(True)


def candidate_ready_for_live(skill: Skill) -> bool:
    """Repeated successful extraction across distinct episodes can promote a candidate."""
    if skill.maturity is not Maturity.CANDIDATE:
        return False
    return len(dict.fromkeys(skill.source_episode_ids)) >= MIN_SOURCE_EPISODES_FOR_LIVE


def _execution_step_count(execution: str) -> int:
    count = 0
    for line in execution.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+\.\s+", stripped) or re.match(r"^[-*]\s+", stripped):
            count += 1
    return count


def _activation_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text)


__all__ = [
    "AdmissionDecision",
    "MIN_EXECUTION_STEPS",
    "MIN_SOURCE_EPISODES_FOR_LIVE",
    "MIN_TOOL_CALLS_FOR_EXTRACTION",
    "admit_extracted_skill",
    "candidate_ready_for_live",
    "should_extract_from_episode",
]
