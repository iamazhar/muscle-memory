"""Tests for the evaluation system."""

from __future__ import annotations

import uuid

import pytest

from muscle_memory.db import Store
from muscle_memory.eval import EvalLabel
from muscle_memory.eval.evaluator import (
    OutcomeEvalResult,
    evaluate_impact,
    evaluate_outcomes,
)
from muscle_memory.models import Episode, Outcome, Scope, ToolCall, Trajectory


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "mm.db"
    return Store(db_path, embedding_dims=4)


def _make_episode(
    outcome: Outcome = Outcome.SUCCESS,
    activated_skills: list[str] | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> Episode:
    return Episode(
        id=uuid.uuid4().hex[:16],
        user_prompt="test prompt",
        trajectory=Trajectory(
            tool_calls=tool_calls
            or [ToolCall(name="Bash", arguments={"command": "ls"}, result="ok")],
        ),
        outcome=outcome,
        reward=0.5 if outcome == Outcome.SUCCESS else -0.5 if outcome == Outcome.FAILURE else 0.0,
        activated_skills=activated_skills or [],
    )


class TestEvalLabels:
    def test_add_and_get_label(self, store):
        label = EvalLabel(
            id="test1",
            label_type="outcome",
            episode_id="ep1",
            heuristic_outcome="success",
            human_outcome="failure",
        )
        store.add_eval_label(label)
        labels = store.get_eval_labels("outcome")
        assert len(labels) == 1
        assert labels[0].human_outcome == "failure"
        assert labels[0].heuristic_outcome == "success"

    def test_get_outcome_label(self, store):
        label = EvalLabel(
            id="test2",
            label_type="outcome",
            episode_id="ep2",
            human_outcome="success",
        )
        store.add_eval_label(label)
        found = store.get_outcome_label("ep2")
        assert found is not None
        assert found.human_outcome == "success"
        assert store.get_outcome_label("nonexistent") is None

    def test_count_labels(self, store):
        assert store.count_eval_labels("outcome") == 0
        store.add_eval_label(EvalLabel(id="a", label_type="outcome", episode_id="ep1"))
        store.add_eval_label(EvalLabel(id="b", label_type="outcome", episode_id="ep2"))
        store.add_eval_label(EvalLabel(id="c", label_type="retrieval", episode_id="ep1"))
        assert store.count_eval_labels("outcome") == 2
        assert store.count_eval_labels("retrieval") == 1

    def test_list_unlabeled_episodes(self, store):
        ep1 = _make_episode()
        ep2 = _make_episode()
        store.add_episode(ep1)
        store.add_episode(ep2)

        unlabeled = store.list_unlabeled_episodes()
        assert len(unlabeled) == 2

        store.add_eval_label(
            EvalLabel(id="lbl1", label_type="outcome", episode_id=ep1.id)
        )
        unlabeled = store.list_unlabeled_episodes()
        assert len(unlabeled) == 1
        assert unlabeled[0].id == ep2.id


class TestOutcomeEval:
    def test_empty_labels(self, store):
        result = evaluate_outcomes(store)
        assert result.total == 0

    def test_agreement(self, store):
        # Use a trajectory the heuristic actually calls SUCCESS
        ep = _make_episode(
            outcome=Outcome.SUCCESS,
            tool_calls=[
                ToolCall(name="Bash", arguments={"command": "pytest"}, result="5 passed in 1.2s"),
            ],
        )
        store.add_episode(ep)
        store.add_eval_label(
            EvalLabel(
                id="l1",
                label_type="outcome",
                episode_id=ep.id,
                human_outcome="success",
            )
        )
        result = evaluate_outcomes(store)
        assert result.total == 1
        assert result.agreement == 1
        assert result.agreement_rate == 1.0

    def test_disagreement(self, store):
        # Heuristic says SUCCESS (pytest passed), human says FAILURE
        ep = _make_episode(
            outcome=Outcome.SUCCESS,
            tool_calls=[
                ToolCall(name="Bash", arguments={"command": "pytest"}, result="5 passed in 1.2s"),
            ],
        )
        store.add_episode(ep)
        store.add_eval_label(
            EvalLabel(
                id="l2",
                label_type="outcome",
                episode_id=ep.id,
                human_outcome="failure",
            )
        )
        result = evaluate_outcomes(store)
        assert result.agreement == 0
        assert len(result.disagreements) == 1

    def test_precision_recall(self):
        result = OutcomeEvalResult(
            total=4,
            agreement=3,
            matrix={
                "success": {"success": 2, "failure": 1, "unknown": 0},
                "failure": {"success": 0, "failure": 1, "unknown": 0},
                "unknown": {"success": 0, "failure": 0, "unknown": 0},
            },
        )
        # precision(success) = 2 / (2+1+0) = 0.667
        assert abs(result.precision("success") - 2 / 3) < 0.01
        # recall(success) = 2 / (2+0+0) = 1.0
        assert result.recall("success") == 1.0
        # precision(failure) = 1 / (1+0) = 1.0
        assert result.precision("failure") == 1.0
        # recall(failure) = 1 / (1+1+0) = 0.5
        assert result.recall("failure") == 0.5


class TestImpactEval:
    def test_with_vs_without(self, store):
        # 2 episodes with skills (1 success, 1 failure)
        store.add_episode(_make_episode(Outcome.SUCCESS, activated_skills=["s1"]))
        store.add_episode(_make_episode(Outcome.FAILURE, activated_skills=["s1"]))
        # 2 episodes without skills (both unknown)
        store.add_episode(_make_episode(Outcome.UNKNOWN))
        store.add_episode(_make_episode(Outcome.UNKNOWN))

        result = evaluate_impact(store)
        assert result.with_skills.count == 2
        assert result.without_skills.count == 2
        assert result.with_skills.success_rate == 0.5
        assert result.without_skills.success_rate == 0.0

    def test_per_skill_requires_3_episodes(self, store):
        # Only 2 episodes for skill "s1" — below threshold
        store.add_episode(_make_episode(Outcome.SUCCESS, activated_skills=["s1"]))
        store.add_episode(_make_episode(Outcome.SUCCESS, activated_skills=["s1"]))
        result = evaluate_impact(store)
        assert len(result.per_skill) == 0

        # Add a third
        store.add_episode(_make_episode(Outcome.FAILURE, activated_skills=["s1"]))
        result = evaluate_impact(store)
        assert len(result.per_skill) == 1
        assert abs(result.per_skill[0].success_rate - 2 / 3) < 0.01
