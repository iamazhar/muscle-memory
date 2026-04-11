"""Tests for the v0.2 Non-Parametric PPO refinement loop.

Uses a FakeLLM that returns canned JSON responses so tests run offline
and deterministically. Verifies the three stages:

  1. extract_gradient → SemanticGradient
  2. apply_gradient → revised Skill with previous_text preserved
  3. verify_refinement → PPO-Gate accept/reject logic

Plus the full refine_skill() orchestrator, trigger criteria, and
rollback via the DB layer.
"""

from __future__ import annotations

from typing import Any

import pytest

from muscle_memory.db import Store
from muscle_memory.models import Episode, Maturity, Outcome, Skill, ToolCall, Trajectory
from muscle_memory.refine import (
    RefinementError,
    SemanticGradient,
    apply_gradient,
    extract_gradient,
    refine_skill,
    should_auto_refine,
    verify_refinement,
)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


class ScriptedLLM:
    """LLM stub that returns a queue of canned JSON responses in order.

    Each call to `complete_json` pops the next response. Raises if the
    queue is empty so missed calls are loud.
    """

    model = "scripted"

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, **_: Any) -> Any:
        self.calls.append((system[:60], user[:60]))
        if not self.responses:
            raise RuntimeError("ScriptedLLM out of canned responses")
        return self.responses.pop(0)

    def complete_text(self, system: str, user: str, **_: Any) -> str:
        return ""


class ExplodingLLM:
    model = "exploding"

    def complete_json(self, *a, **k):
        raise RuntimeError("kaboom")

    def complete_text(self, *a, **k):
        raise RuntimeError("kaboom")


def _bug_skill() -> Skill:
    """The kind of skill our real dogfood produced — broken python path."""
    return Skill(
        id="test_bug_0001",
        activation="When pytest fails with ModuleNotFoundError on macOS after uv sync",
        execution=(
            "1. Run `ls -lO .venv/lib/python*/site-packages/*.pth`\n"
            "2. Run `chflags nohidden .venv/lib/python*/site-packages/*.pth`\n"
            "3. Verify with `python -c 'import pkg'`"
        ),
        termination="import succeeds",
        tags=["macos", "uv"],
        invocations=6,
        successes=2,
        failures=3,
        score=0.33,
        maturity=Maturity.CANDIDATE,
    )


def _failing_episode(ep_id: str = "ep-1") -> Episode:
    return Episode(
        id=ep_id,
        user_prompt="test the uv pth probe",
        trajectory=Trajectory(
            tool_calls=[
                ToolCall(
                    name="Bash",
                    arguments={"command": "ls -lO .venv/lib/python*/site-packages/*.pth"},
                    result="-rw-r--r--@ hidden .pth",
                ),
                ToolCall(
                    name="Bash",
                    arguments={"command": "chflags nohidden ..."},
                    result="",
                ),
                ToolCall(
                    name="Bash",
                    arguments={"command": "python -c 'import pkg'"},
                    error="command not found: python",
                ),
                ToolCall(
                    name="Bash",
                    arguments={"command": "python3 -c 'import pkg'"},
                    result="/path/to/pkg/__init__.py",
                ),
            ]
        ),
        outcome=Outcome.SUCCESS,
        reward=0.6,
        activated_skills=["test_bug_0001"],
    )


# ----------------------------------------------------------------------
# Stage 1 — extract_gradient
# ----------------------------------------------------------------------


class TestExtractGradient:
    def test_happy_path_parses_canned_response(self) -> None:
        llm = ScriptedLLM(
            [
                {
                    "field_feedback": {
                        "activation": "No change",
                        "execution": "Step 3 uses `python` but macOS needs `python3`. Change to `.venv/bin/python3`.",
                        "termination": "No change",
                    },
                    "root_cause": "execution step 3 uses the wrong interpreter on macOS",
                    "suggested_intent": "use python3 explicitly in the verify step",
                    "severity": "minor",
                    "should_refine": True,
                }
            ]
        )
        gradient = extract_gradient(_bug_skill(), [_failing_episode()], llm)
        assert gradient.should_refine is True
        assert gradient.severity == "minor"
        assert "python3" in gradient.field_feedback["execution"]
        assert gradient.field_feedback["activation"] == "No change"
        assert gradient.any_changes() is True

    def test_no_change_gradient_has_no_changes(self) -> None:
        llm = ScriptedLLM(
            [
                {
                    "field_feedback": {
                        "activation": "No change",
                        "execution": "No change",
                        "termination": "No change",
                    },
                    "root_cause": "nothing wrong",
                    "suggested_intent": "no revision needed",
                    "severity": "minor",
                    "should_refine": False,
                }
            ]
        )
        gradient = extract_gradient(_bug_skill(), [_failing_episode()], llm)
        assert gradient.should_refine is False
        assert gradient.any_changes() is False

    def test_llm_failure_raises_refinement_error(self) -> None:
        with pytest.raises(RefinementError, match="gradient extraction"):
            extract_gradient(_bug_skill(), [_failing_episode()], ExplodingLLM())

    def test_requires_at_least_one_episode(self) -> None:
        with pytest.raises(RefinementError, match="at least one episode"):
            extract_gradient(_bug_skill(), [], ScriptedLLM([{}]))

    def test_malformed_response_raises(self) -> None:
        llm = ScriptedLLM(["this is not a dict"])
        with pytest.raises(RefinementError):
            extract_gradient(_bug_skill(), [_failing_episode()], llm)


# ----------------------------------------------------------------------
# Stage 2 — apply_gradient
# ----------------------------------------------------------------------


class TestApplyGradient:
    def test_rewrite_preserves_unchanged_fields_and_bumps_count(self) -> None:
        skill = _bug_skill()
        gradient = SemanticGradient(
            field_feedback={
                "activation": "No change",
                "execution": "Use python3 in step 3",
                "termination": "No change",
            },
            root_cause="python vs python3",
            suggested_intent="use python3",
            severity="minor",
            should_refine=True,
        )
        llm = ScriptedLLM(
            [
                {
                    "activation": skill.activation,
                    "execution": (
                        "1. Run `ls -lO .venv/lib/python*/site-packages/*.pth`\n"
                        "2. Run `chflags nohidden .venv/lib/python*/site-packages/*.pth`\n"
                        "3. Verify with `.venv/bin/python3 -c 'import pkg'`"
                    ),
                    "termination": skill.termination,
                }
            ]
        )
        revised = apply_gradient(skill, gradient, llm)

        assert revised.id == skill.id  # same id
        assert revised.activation == skill.activation  # unchanged
        assert "python3" in revised.execution  # changed
        assert revised.termination == skill.termination  # unchanged
        assert revised.refinement_count == skill.refinement_count + 1
        assert revised.previous_text is not None
        assert revised.previous_text["execution"] == skill.execution
        assert revised.last_refined_at is not None

        # scoring state carries over
        assert revised.successes == skill.successes
        assert revised.invocations == skill.invocations

    def test_rewrite_llm_failure_raises(self) -> None:
        with pytest.raises(RefinementError, match="rewrite"):
            apply_gradient(
                _bug_skill(),
                SemanticGradient(
                    field_feedback={"activation": "", "execution": "", "termination": ""},
                    root_cause="x",
                    suggested_intent="y",
                    severity="minor",
                    should_refine=True,
                ),
                ExplodingLLM(),
            )

    def test_rewrite_with_empty_response_falls_back_to_original(self) -> None:
        """If the LLM returns empty strings, the revised skill should
        fall back to the original field values (never empty)."""
        skill = _bug_skill()
        gradient = SemanticGradient(
            field_feedback={"activation": "", "execution": "", "termination": ""},
            root_cause="x",
            suggested_intent="y",
            severity="minor",
            should_refine=True,
        )
        llm = ScriptedLLM([{}])
        revised = apply_gradient(skill, gradient, llm)
        # falls back to original (Skill validator would reject empties)
        assert revised.activation == skill.activation
        assert revised.execution == skill.execution
        assert revised.termination == skill.termination


# ----------------------------------------------------------------------
# Stage 3 — PPO Gate verification
# ----------------------------------------------------------------------


class TestVerifyRefinement:
    def _make_original_and_revised(self) -> tuple[Skill, Skill]:
        original = _bug_skill()
        revised = Skill(
            id=original.id,
            activation=original.activation,
            execution=original.execution.replace("python", "python3"),
            termination=original.termination,
            successes=original.successes,
            invocations=original.invocations,
            failures=original.failures,
            maturity=original.maturity,
            source_episode_ids=list(original.source_episode_ids),
            refinement_count=original.refinement_count + 1,
            previous_text={
                "activation": original.activation,
                "execution": original.execution,
                "termination": original.termination,
            },
        )
        return original, revised

    def test_accepts_on_positive_mean(self) -> None:
        original, revised = self._make_original_and_revised()
        llm = ScriptedLLM(
            [
                {"score": 2, "reason": "fixes the exact bug"},
                {"score": 1, "reason": "slight improvement"},
                {"score": 0, "reason": "unaffected"},
            ]
        )
        accepted, verdicts, reason = verify_refinement(
            original,
            revised,
            [_failing_episode("e1"), _failing_episode("e2"), _failing_episode("e3")],
            llm,
        )
        assert accepted is True
        assert reason is None
        assert len(verdicts) == 3

    def test_rejects_on_strong_regression_veto(self) -> None:
        original, revised = self._make_original_and_revised()
        llm = ScriptedLLM(
            [
                {"score": 2, "reason": "good"},
                {"score": -2, "reason": "broke something important"},
                {"score": 1, "reason": "ok"},
            ]
        )
        accepted, _verdicts, reason = verify_refinement(
            original,
            revised,
            [_failing_episode("e1"), _failing_episode("e2"), _failing_episode("e3")],
            llm,
        )
        assert accepted is False
        assert "strong regression" in reason

    def test_rejects_on_low_mean(self) -> None:
        original, revised = self._make_original_and_revised()
        llm = ScriptedLLM(
            [
                {"score": 0, "reason": "neutral"},
                {"score": 0, "reason": "neutral"},
                {"score": 0, "reason": "neutral"},
            ]
        )
        accepted, _verdicts, reason = verify_refinement(
            original,
            revised,
            [_failing_episode("e1"), _failing_episode("e2"), _failing_episode("e3")],
            llm,
        )
        assert accepted is False
        assert "mean judge score" in reason

    def test_rejects_when_no_text_changes(self) -> None:
        original = _bug_skill()
        # revised is identical — should be rejected before any LLM call
        revised = Skill(
            id=original.id,
            activation=original.activation,
            execution=original.execution,
            termination=original.termination,
        )
        accepted, verdicts, reason = verify_refinement(
            original, revised, [_failing_episode()], ScriptedLLM([])
        )
        assert accepted is False
        assert "no textual changes" in reason
        assert verdicts == []

    def test_rejects_when_no_trajectories_available(self) -> None:
        original, revised = self._make_original_and_revised()
        accepted, _v, reason = verify_refinement(original, revised, [], ScriptedLLM([]))
        assert accepted is False
        assert "no trajectories" in reason

    def test_judge_error_counts_as_neutral(self) -> None:
        original, revised = self._make_original_and_revised()

        # first call succeeds +2, second crashes, third succeeds +1
        # (exploding LLM isn't compatible with ScriptedLLM so we build inline)
        class HalfExploding:
            model = "half"

            def __init__(self):
                self.calls = 0

            def complete_json(self, *a, **k):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("mid-judge failure")
                return {"score": 2, "reason": "works"}

            def complete_text(self, *a, **k):
                return ""

        accepted, verdicts, _reason = verify_refinement(
            original,
            revised,
            [_failing_episode("e1"), _failing_episode("e2"), _failing_episode("e3")],
            HalfExploding(),
        )
        # 3 verdicts: +2, 0 (neutral on error), +2 → mean 1.33 → accept
        assert len(verdicts) == 3
        assert any("judge failed" in v.reason for v in verdicts)
        assert accepted is True


# ----------------------------------------------------------------------
# Orchestrator — refine_skill
# ----------------------------------------------------------------------


class TestRefineSkillOrchestrator:
    def test_full_accept_path_persists_to_store(self, tmp_db: Store) -> None:
        skill = _bug_skill()
        tmp_db.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])

        llm = ScriptedLLM(
            [
                # stage 1: gradient
                {
                    "field_feedback": {
                        "activation": "No change",
                        "execution": "Use python3 instead of python in step 3",
                        "termination": "No change",
                    },
                    "root_cause": "python vs python3",
                    "suggested_intent": "use python3",
                    "severity": "minor",
                    "should_refine": True,
                },
                # stage 2: rewrite
                {
                    "activation": skill.activation,
                    "execution": skill.execution.replace("python", "python3"),
                    "termination": skill.termination,
                },
                # stage 3: judge (3 verdicts, all positive)
                {"score": 2, "reason": "fixes bug"},
                {"score": 1, "reason": "helps"},
                {"score": 1, "reason": "helps"},
            ]
        )

        episodes = [_failing_episode(f"ep{i}") for i in range(3)]
        result = refine_skill(skill, episodes, llm, store=tmp_db)
        assert result.accepted is True
        assert result.revised_skill is not None

        # persisted to store
        reloaded = tmp_db.get_skill(skill.id)
        assert reloaded is not None
        assert "python3" in reloaded.execution
        assert reloaded.refinement_count == 1
        assert reloaded.previous_text is not None
        assert reloaded.previous_text["execution"] == skill.execution

    def test_gradient_says_no_refine_returns_early(self, tmp_db: Store) -> None:
        skill = _bug_skill()
        tmp_db.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])
        llm = ScriptedLLM(
            [
                {
                    "field_feedback": {
                        "activation": "No change",
                        "execution": "No change",
                        "termination": "No change",
                    },
                    "root_cause": "nothing wrong",
                    "suggested_intent": "no revision",
                    "severity": "minor",
                    "should_refine": False,
                }
            ]
        )
        result = refine_skill(skill, [_failing_episode()], llm, store=tmp_db)
        assert result.accepted is False
        assert "should_refine=false" in result.rejection_reason
        # no changes persisted
        reloaded = tmp_db.get_skill(skill.id)
        assert reloaded.execution == skill.execution

    def test_ppo_gate_rejection_does_not_persist(self, tmp_db: Store) -> None:
        skill = _bug_skill()
        tmp_db.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])
        llm = ScriptedLLM(
            [
                {
                    "field_feedback": {
                        "activation": "No change",
                        "execution": "Change it",
                        "termination": "No change",
                    },
                    "root_cause": "x",
                    "suggested_intent": "y",
                    "severity": "minor",
                    "should_refine": True,
                },
                {
                    "activation": skill.activation,
                    "execution": "totally different execution that regresses things",
                    "termination": skill.termination,
                },
                # judges all score 0 → mean 0 → rejected (< 0.5 threshold)
                {"score": 0, "reason": "neutral"},
                {"score": 0, "reason": "neutral"},
            ]
        )
        result = refine_skill(
            skill,
            [_failing_episode("a"), _failing_episode("b")],
            llm,
            store=tmp_db,
        )
        assert result.accepted is False
        assert "mean judge score" in result.rejection_reason
        reloaded = tmp_db.get_skill(skill.id)
        assert reloaded.execution == skill.execution  # unchanged
        assert reloaded.refinement_count == 0

    def test_gradient_failure_never_raises(self) -> None:
        skill = _bug_skill()
        result = refine_skill(skill, [_failing_episode()], ExplodingLLM())
        assert result.accepted is False
        assert "gradient extraction failed" in result.rejection_reason


# ----------------------------------------------------------------------
# Trigger criteria — should_auto_refine
# ----------------------------------------------------------------------


class TestAutoRefineTrigger:
    def test_fires_for_low_scoring_skill_with_enough_data(self) -> None:
        skill = Skill(
            activation="x",
            execution="y",
            termination="z",
            maturity=Maturity.LIVE,
            invocations=10,
            successes=3,
            failures=7,
            score=0.3,
        )
        assert should_auto_refine(skill)

    def test_does_not_fire_for_high_success_rate(self) -> None:
        skill = Skill(
            activation="x",
            execution="y",
            termination="z",
            maturity=Maturity.LIVE,
            invocations=10,
            successes=9,
            failures=1,
            score=0.9,
        )
        assert not should_auto_refine(skill)

    def test_does_not_fire_without_enough_invocations(self) -> None:
        skill = Skill(
            activation="x",
            execution="y",
            termination="z",
            maturity=Maturity.LIVE,
            invocations=2,
            successes=0,
            failures=2,
            score=0.0,
        )
        assert not should_auto_refine(skill)

    def test_does_not_fire_without_failures(self) -> None:
        skill = Skill(
            activation="x",
            execution="y",
            termination="z",
            maturity=Maturity.LIVE,
            invocations=10,
            successes=5,
            failures=0,
            score=0.5,
        )
        assert not should_auto_refine(skill)

    def test_does_not_fire_for_quarantined_candidate(self) -> None:
        skill = Skill(
            activation="x",
            execution="y",
            termination="z",
            maturity=Maturity.CANDIDATE,
            invocations=10,
            successes=3,
            failures=7,
            score=0.3,
        )
        assert not should_auto_refine(skill)
