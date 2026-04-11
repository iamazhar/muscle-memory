"""Tests for the evaluation system."""

from __future__ import annotations

import uuid

import pytest

from muscle_memory.db import Store
from muscle_memory.eval import EvalLabel
from muscle_memory.eval.evaluator import (
    evaluate_credits,
    evaluate_impact,
)
from muscle_memory.models import Episode, Outcome, Skill, ToolCall, Trajectory


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


def _make_skill(store, skill_id: str = "") -> Skill:
    sid = skill_id or uuid.uuid4().hex[:16]
    skill = Skill(
        id=sid,
        activation="When X happens",
        execution="Do Y",
        termination="Done when Z",
    )
    store.add_skill(skill)
    return skill


class TestEvalLabels:
    def test_add_and_get_label(self, store):
        label = EvalLabel(
            id="test1",
            label_type="credit",
            episode_id="ep1",
            skill_id="sk1",
            heuristic_outcome="success",
            human_outcome="deserved",
        )
        store.add_eval_label(label)
        labels = store.get_eval_labels("credit")
        assert len(labels) == 1
        assert labels[0].human_outcome == "deserved"

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
        assert store.count_eval_labels("credit") == 0
        store.add_eval_label(
            EvalLabel(id="a", label_type="credit", episode_id="ep1", skill_id="s1")
        )
        store.add_eval_label(
            EvalLabel(id="b", label_type="credit", episode_id="ep2", skill_id="s2")
        )
        store.add_eval_label(EvalLabel(id="c", label_type="outcome", episode_id="ep1"))
        assert store.count_eval_labels("credit") == 2
        assert store.count_eval_labels("outcome") == 1

    def test_list_unlabeled_episodes(self, store):
        ep1 = _make_episode()
        ep2 = _make_episode()
        store.add_episode(ep1)
        store.add_episode(ep2)

        unlabeled = store.list_unlabeled_episodes()
        assert len(unlabeled) == 2

        store.add_eval_label(EvalLabel(id="lbl1", label_type="outcome", episode_id=ep1.id))
        unlabeled = store.list_unlabeled_episodes()
        assert len(unlabeled) == 1
        assert unlabeled[0].id == ep2.id


class TestCreditEval:
    def test_empty_labels(self, store):
        result = evaluate_credits(store)
        assert result.total == 0

    def test_all_deserved(self, store):
        skill = _make_skill(store)
        ep = _make_episode(activated_skills=[skill.id])
        store.add_episode(ep)

        store.add_eval_label(
            EvalLabel(
                id="l1",
                label_type="credit",
                episode_id=ep.id,
                skill_id=skill.id,
                heuristic_outcome="success",
                human_outcome="deserved",
            )
        )
        result = evaluate_credits(store)
        assert result.total == 1
        assert result.deserved == 1
        assert result.precision == 1.0

    def test_mixed_credits(self, store):
        s1 = _make_skill(store, "skill_good")
        s2 = _make_skill(store, "skill_bad_")
        ep = _make_episode(activated_skills=[s1.id, s2.id])
        store.add_episode(ep)

        store.add_eval_label(
            EvalLabel(
                id="l1",
                label_type="credit",
                episode_id=ep.id,
                skill_id=s1.id,
                human_outcome="deserved",
            )
        )
        store.add_eval_label(
            EvalLabel(
                id="l2",
                label_type="credit",
                episode_id=ep.id,
                skill_id=s2.id,
                human_outcome="undeserved",
            )
        )
        result = evaluate_credits(store)
        assert result.total == 2
        assert result.deserved == 1
        assert result.undeserved == 1
        assert result.precision == 0.5

    def test_per_skill_breakdown(self, store):
        s1 = _make_skill(store, "skill_good2")
        ep1 = _make_episode(activated_skills=[s1.id])
        ep2 = _make_episode(activated_skills=[s1.id])
        store.add_episode(ep1)
        store.add_episode(ep2)

        store.add_eval_label(
            EvalLabel(
                id="l1",
                label_type="credit",
                episode_id=ep1.id,
                skill_id=s1.id,
                human_outcome="deserved",
            )
        )
        store.add_eval_label(
            EvalLabel(
                id="l2",
                label_type="credit",
                episode_id=ep2.id,
                skill_id=s1.id,
                human_outcome="undeserved",
            )
        )
        result = evaluate_credits(store)
        assert len(result.per_skill) == 1
        assert result.per_skill[0].precision == 0.5


class TestImpactEval:
    def test_with_vs_without(self, store):
        store.add_episode(_make_episode(Outcome.SUCCESS, activated_skills=["s1"]))
        store.add_episode(_make_episode(Outcome.FAILURE, activated_skills=["s1"]))
        store.add_episode(_make_episode(Outcome.UNKNOWN))
        store.add_episode(_make_episode(Outcome.UNKNOWN))

        result = evaluate_impact(store)
        assert result.with_skills.count == 2
        assert result.without_skills.count == 2
        assert result.with_skills.success_rate == 0.5
        assert result.without_skills.success_rate == 0.0

    def test_per_skill_requires_3_episodes(self, store):
        store.add_episode(_make_episode(Outcome.SUCCESS, activated_skills=["s1"]))
        store.add_episode(_make_episode(Outcome.SUCCESS, activated_skills=["s1"]))
        result = evaluate_impact(store)
        assert len(result.per_skill) == 0

        store.add_episode(_make_episode(Outcome.FAILURE, activated_skills=["s1"]))
        result = evaluate_impact(store)
        assert len(result.per_skill) == 1
        assert abs(result.per_skill[0].success_rate - 2 / 3) < 0.01
