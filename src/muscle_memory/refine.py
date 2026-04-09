"""Non-Parametric PPO refinement loop for muscle-memory skills.

Ports the core idea of ProcMEM's (arxiv:2602.01869) non-parametric PPO
mechanism into the Claude Code coding-agent setting. The paper's
three stages are:

  1. **Semantic gradient extraction** — analyze a skill + its recent
     trajectories and produce per-field feedback describing what's
     wrong and why.
  2. **Skill text rewriting** — apply the gradient via a second LLM
     call that edits the skill text in place.
  3. **PPO Gate verification** — decide whether the revised skill is
     actually better than the original on the evidence of stored
     trajectories. Accept only if it is.

The paper computes the PPO importance ratio from token-level
log-probabilities, which the Anthropic API does not expose. Instead we
use an **LLM-judge proxy**: for each trajectory, ask Sonnet to score
the proposed revision on a -2..+2 scale relative to the original.
Aggregate across trajectories; accept only if the mean is clearly
positive AND no single trajectory strongly regresses. This is the
spirit of the trust-region constraint adapted to a closed-source LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from typing import Any

from muscle_memory.db import Store
from muscle_memory.extractor import _short
from muscle_memory.llm import LLM
from muscle_memory.models import Episode, Skill


class RefinementError(RuntimeError):
    """LLM call failed during refinement. Callers decide whether to
    propagate or swallow. Unlike `ExtractionError`, a refinement failure
    should never modify the skill — on error we keep the original."""


@dataclass
class SemanticGradient:
    """Per-field feedback produced by Stage 1 of refinement."""

    field_feedback: dict[str, str]  # keys: activation / execution / termination
    root_cause: str
    suggested_intent: str
    severity: str  # minor | moderate | major
    should_refine: bool

    def any_changes(self) -> bool:
        """Is there at least one non-trivial field-level change?"""
        if not self.should_refine:
            return False
        for value in self.field_feedback.values():
            if value.strip().lower() not in {"", "no change", "no change."}:
                return True
        return False


@dataclass
class JudgeVerdict:
    episode_id: str
    score: int  # -2..+2
    reason: str


@dataclass
class RefinementResult:
    """Outcome of a refinement attempt on a single skill."""

    skill_id: str
    accepted: bool
    gradient: SemanticGradient | None = None
    revised_skill: Skill | None = None
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    mean_judge_score: float = 0.0
    rejection_reason: str | None = None

    def summary(self) -> str:
        if self.accepted:
            return (
                f"ACCEPTED refinement of {self.skill_id[:8]}: "
                f"mean judge score {self.mean_judge_score:+.2f} across "
                f"{len(self.verdicts)} trajectories"
            )
        if self.rejection_reason:
            return f"REJECTED refinement of {self.skill_id[:8]}: {self.rejection_reason}"
        return f"REJECTED refinement of {self.skill_id[:8]} (no reason recorded)"


# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    return resources.files("muscle_memory.prompts").joinpath(name).read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Stage 1 — semantic gradient extraction
# ----------------------------------------------------------------------


def _format_skill_block(skill: Skill) -> str:
    lines = [
        "<skill>",
        f"  <id>{skill.id}</id>",
        f"  <refinement_count>{skill.refinement_count}</refinement_count>",
        f"  <invocations>{skill.invocations}</invocations>",
        f"  <successes>{skill.successes}</successes>",
        f"  <failures>{skill.failures}</failures>",
        f"  <score>{skill.score:.2f}</score>",
        "  <activation>",
        skill.activation,
        "  </activation>",
        "  <execution>",
        skill.execution,
        "  </execution>",
        "  <termination>",
        skill.termination,
        "  </termination>",
        "</skill>",
    ]
    return "\n".join(lines)


def _format_episode_block(episode: Episode, *, index: int) -> str:
    lines = [
        f'<episode index="{index}" id="{episode.id[:8]}" outcome="{episode.outcome.value}">',
        f"  <user_prompt>{_short(episode.user_prompt, 300)}</user_prompt>",
    ]
    if episode.reward != 0:
        lines.append(f"  <reward>{episode.reward:+.2f}</reward>")
    # flatten trajectory — reuse the extractor's formatter, which is
    # already tuned for LLM readability
    lines.append("  <trajectory_summary>")
    # shorter view inline — the extractor's formatter is too verbose for refinement
    for i, tc in enumerate(episode.trajectory.tool_calls[:25]):
        parts = [f"    [{i}] {tc.name}"]
        if tc.arguments:
            parts.append(
                "       args: " + ", ".join(f"{k}={_short(v, 80)}" for k, v in tc.arguments.items())
            )
        if tc.result:
            parts.append(f"       result: {_short(tc.result, 150)}")
        if tc.error:
            parts.append(f"       ERROR: {_short(tc.error, 150)}")
        lines.append("\n".join(parts))
    if len(episode.trajectory.tool_calls) > 25:
        lines.append(f"    ... ({len(episode.trajectory.tool_calls) - 25} more tool calls elided)")
    lines.append("  </trajectory_summary>")
    lines.append("</episode>")
    return "\n".join(lines)


def extract_gradient(
    skill: Skill,
    episodes: list[Episode],
    llm: LLM,
) -> SemanticGradient:
    """Stage 1: produce per-field feedback via an LLM call.

    Raises RefinementError on any LLM/parse failure.
    """
    if not episodes:
        raise RefinementError("extract_gradient needs at least one episode")

    system = _load_prompt("refine_gradient.md")

    user_lines = [_format_skill_block(skill), "", "<episodes>"]
    for i, ep in enumerate(episodes):
        user_lines.append(_format_episode_block(ep, index=i))
    user_lines.append("</episodes>")
    user_lines.append("")
    user_lines.append("Now produce your JSON diagnosis.")
    user = "\n".join(user_lines)

    try:
        raw = llm.complete_json(system, user, max_tokens=2048, temperature=0.2)
    except Exception as exc:
        raise RefinementError(f"gradient extraction LLM call failed: {exc}") from exc

    return _coerce_gradient(raw)


def _coerce_gradient(raw: Any) -> SemanticGradient:
    if not isinstance(raw, dict):
        raise RefinementError(f"gradient response must be an object, got {type(raw).__name__}")
    field_feedback = raw.get("field_feedback")
    if not isinstance(field_feedback, dict):
        raise RefinementError("gradient missing 'field_feedback' object")

    # normalize keys
    normalized: dict[str, str] = {}
    for k in ("activation", "execution", "termination"):
        v = field_feedback.get(k, "No change")
        normalized[k] = str(v) if v is not None else "No change"

    severity = str(raw.get("severity", "minor")).lower()
    if severity not in {"minor", "moderate", "major"}:
        severity = "minor"

    return SemanticGradient(
        field_feedback=normalized,
        root_cause=str(raw.get("root_cause", "")),
        suggested_intent=str(raw.get("suggested_intent", "")),
        severity=severity,
        should_refine=bool(raw.get("should_refine", True)),
    )


# ----------------------------------------------------------------------
# Stage 2 — apply the gradient (rewrite)
# ----------------------------------------------------------------------


def apply_gradient(
    skill: Skill,
    gradient: SemanticGradient,
    llm: LLM,
) -> Skill:
    """Stage 2: produce a candidate refined skill via an LLM rewrite.

    Returns a NEW Skill object with the same id but revised text. The
    caller is responsible for verification (Stage 3) before persisting
    the change to the store.
    """
    system = _load_prompt("refine_rewrite.md")

    user_lines = [
        "<original_skill>",
        f"  <activation>{skill.activation}</activation>",
        f"  <execution>{skill.execution}</execution>",
        f"  <termination>{skill.termination}</termination>",
        "</original_skill>",
        "",
        "<gradient>",
        json.dumps(
            {
                "field_feedback": gradient.field_feedback,
                "root_cause": gradient.root_cause,
                "suggested_intent": gradient.suggested_intent,
                "severity": gradient.severity,
            },
            indent=2,
        ),
        "</gradient>",
        "",
        f"<intent>{gradient.suggested_intent}</intent>",
        "",
        "Now produce the revised JSON object with the three fields.",
    ]
    user = "\n".join(user_lines)

    try:
        raw = llm.complete_json(system, user, max_tokens=2048, temperature=0.2)
    except Exception as exc:
        raise RefinementError(f"rewrite LLM call failed: {exc}") from exc

    if not isinstance(raw, dict):
        raise RefinementError("rewrite response must be a JSON object")

    try:
        revised = Skill(
            id=skill.id,
            activation=str(raw.get("activation", skill.activation)),
            execution=str(raw.get("execution", skill.execution)),
            termination=str(raw.get("termination", skill.termination)),
            tool_hints=list(skill.tool_hints),
            tags=list(skill.tags),
            scope=skill.scope,
            score=skill.score,
            invocations=skill.invocations,
            successes=skill.successes,
            failures=skill.failures,
            maturity=skill.maturity,
            source_episode_ids=list(skill.source_episode_ids),
            refinement_count=skill.refinement_count + 1,
            previous_text={
                "activation": skill.activation,
                "execution": skill.execution,
                "termination": skill.termination,
            },
            created_at=skill.created_at,
            last_used_at=skill.last_used_at,
            last_refined_at=datetime.now(UTC),
        )
    except Exception as exc:
        raise RefinementError(f"rewrite produced invalid skill: {exc}") from exc

    return revised


# ----------------------------------------------------------------------
# Stage 3 — PPO Gate verification (LLM-judge proxy)
# ----------------------------------------------------------------------


def _diff_summary(old: Skill, new: Skill) -> str:
    changed: list[str] = []
    if old.activation != new.activation:
        changed.append("activation changed")
    if old.execution != new.execution:
        changed.append("execution changed")
    if old.termination != new.termination:
        changed.append("termination changed")
    if not changed:
        return "no textual changes"
    return "; ".join(changed)


def judge_one_trajectory(
    original: Skill,
    revised: Skill,
    episode: Episode,
    llm: LLM,
) -> JudgeVerdict:
    """Ask the LLM to score one trajectory: would the revised skill
    have done better, same, or worse?
    """
    system = _load_prompt("refine_judge.md")

    user_lines = [
        "<original_skill>",
        f"  <activation>{original.activation}</activation>",
        f"  <execution>{original.execution}</execution>",
        f"  <termination>{original.termination}</termination>",
        "</original_skill>",
        "",
        "<revised_skill>",
        f"  <activation>{revised.activation}</activation>",
        f"  <execution>{revised.execution}</execution>",
        f"  <termination>{revised.termination}</termination>",
        "</revised_skill>",
        "",
        _format_episode_block(episode, index=0),
        "",
        f"<diff_summary>{_diff_summary(original, revised)}</diff_summary>",
        "",
        "Now produce your JSON verdict.",
    ]
    user = "\n".join(user_lines)

    try:
        raw = llm.complete_json(system, user, max_tokens=512, temperature=0.1)
    except Exception as exc:
        raise RefinementError(f"judge LLM call failed: {exc}") from exc

    if not isinstance(raw, dict):
        raise RefinementError("judge response must be an object")

    try:
        score_raw = raw.get("score", 0)
        score = max(-2, min(2, int(score_raw)))
    except (TypeError, ValueError):
        score = 0

    return JudgeVerdict(
        episode_id=episode.id,
        score=score,
        reason=str(raw.get("reason", "")),
    )


# PPO-Gate thresholds. Calibrated conservatively: we reject refinements
# unless the evidence is clearly positive.
ACCEPT_MEAN_THRESHOLD = 0.5  # mean judge score across trajectories
STRONG_REGRESSION_BAR = -2  # any single verdict at -2 is a veto


def verify_refinement(
    original: Skill,
    revised: Skill,
    episodes: list[Episode],
    llm: LLM,
) -> tuple[bool, list[JudgeVerdict], str | None]:
    """Stage 3: judge the revision across stored trajectories.

    Returns (accepted, verdicts, rejection_reason).
    """
    if _diff_summary(original, revised) == "no textual changes":
        return False, [], "rewrite produced no textual changes"
    if not episodes:
        # no evidence available — refuse to accept on faith
        return False, [], "no trajectories available for verification"

    verdicts: list[JudgeVerdict] = []
    for ep in episodes:
        try:
            verdict = judge_one_trajectory(original, revised, ep, llm)
        except RefinementError as exc:
            # keep going — judge errors are treated as neutral verdicts
            verdicts.append(JudgeVerdict(episode_id=ep.id, score=0, reason=f"judge failed: {exc}"))
            continue
        verdicts.append(verdict)

    if not verdicts:
        return False, verdicts, "no judge verdicts returned"

    mean_score = sum(v.score for v in verdicts) / len(verdicts)

    # Trust-region check: a single strong regression vetoes acceptance
    if any(v.score <= STRONG_REGRESSION_BAR for v in verdicts):
        return (
            False,
            verdicts,
            f"strong regression on episode "
            f"{next(v.episode_id[:8] for v in verdicts if v.score <= STRONG_REGRESSION_BAR)}",
        )

    if mean_score < ACCEPT_MEAN_THRESHOLD:
        return (
            False,
            verdicts,
            f"mean judge score {mean_score:+.2f} < threshold {ACCEPT_MEAN_THRESHOLD}",
        )

    return True, verdicts, None


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------


def refine_skill(
    skill: Skill,
    episodes: list[Episode],
    llm: LLM,
    *,
    store: Store | None = None,
) -> RefinementResult:
    """End-to-end refinement of one skill.

    If `store` is provided and the refinement is accepted, persists
    the revised skill via `store.update_skill`. Otherwise the caller
    is responsible for persistence (useful for dry-run / testing).

    Never raises on failure — returns a RefinementResult with
    accepted=False and a populated rejection_reason.
    """
    result = RefinementResult(skill_id=skill.id, accepted=False)

    try:
        gradient = extract_gradient(skill, episodes, llm)
    except RefinementError as exc:
        result.rejection_reason = f"gradient extraction failed: {exc}"
        return result
    result.gradient = gradient

    if not gradient.should_refine:
        result.rejection_reason = "gradient says should_refine=false"
        return result

    if not gradient.any_changes():
        result.rejection_reason = "gradient had no concrete field changes"
        return result

    try:
        revised = apply_gradient(skill, gradient, llm)
    except RefinementError as exc:
        result.rejection_reason = f"rewrite failed: {exc}"
        return result
    result.revised_skill = revised

    accepted, verdicts, reason = verify_refinement(skill, revised, episodes, llm)
    result.verdicts = verdicts
    if verdicts:
        result.mean_judge_score = sum(v.score for v in verdicts) / len(verdicts)
    result.accepted = accepted
    if not accepted:
        result.rejection_reason = reason or "PPO-Gate rejected"
        return result

    # persist if caller provided a store
    if store is not None:
        store.update_skill(revised)

    return result


# ----------------------------------------------------------------------
# Trigger criteria
# ----------------------------------------------------------------------

# Auto-refine thresholds — calibrated so that only skills with real
# failure signal get refinement budget.
MIN_INVOCATIONS_FOR_AUTO = 5
MAX_SUCCESS_RATE_FOR_AUTO = 0.6
MIN_FAILURES_FOR_AUTO = 2


def should_auto_refine(skill: Skill) -> bool:
    """Does this skill meet the criteria for automatic refinement?

    We only auto-refine skills that have:
      - Been tried enough to have signal (≥ MIN_INVOCATIONS_FOR_AUTO)
      - Failed a meaningful number of times (≥ MIN_FAILURES_FOR_AUTO)
      - A success rate below MAX_SUCCESS_RATE_FOR_AUTO
    """
    if skill.invocations < MIN_INVOCATIONS_FOR_AUTO:
        return False
    if skill.failures < MIN_FAILURES_FOR_AUTO:
        return False
    if skill.score > MAX_SUCCESS_RATE_FOR_AUTO:
        return False
    return True


__all__ = [
    "JudgeVerdict",
    "RefinementError",
    "RefinementResult",
    "SemanticGradient",
    "apply_gradient",
    "extract_gradient",
    "judge_one_trajectory",
    "refine_skill",
    "should_auto_refine",
    "verify_refinement",
]
