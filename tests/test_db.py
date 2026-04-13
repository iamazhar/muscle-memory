"""Tests for muscle_memory.db — CRUD and vector search."""

from __future__ import annotations

from muscle_memory.db import Store
from muscle_memory.models import Episode, Maturity, Outcome, Scope, Skill


def test_add_and_get_skill(tmp_db: Store, sample_skill: Skill) -> None:
    tmp_db.add_skill(sample_skill, embedding=[0.1, 0.2, 0.3, 0.4])
    loaded = tmp_db.get_skill(sample_skill.id)
    assert loaded is not None
    assert loaded.activation == sample_skill.activation
    assert loaded.execution == sample_skill.execution
    assert loaded.tool_hints == sample_skill.tool_hints
    assert loaded.tags == sample_skill.tags


def test_update_skill(tmp_db: Store, sample_skill: Skill) -> None:
    tmp_db.add_skill(sample_skill, embedding=[0.1, 0.2, 0.3, 0.4])

    sample_skill.invocations = 5
    sample_skill.successes = 4
    sample_skill.source_episode_ids = ["ep1", "ep2"]
    sample_skill.recompute_score()
    sample_skill.recompute_maturity()
    tmp_db.update_skill(sample_skill)

    loaded = tmp_db.get_skill(sample_skill.id)
    assert loaded is not None
    assert loaded.invocations == 5
    assert loaded.successes == 4
    assert loaded.score == 0.8
    assert loaded.maturity is Maturity.ESTABLISHED


def test_delete_skill(tmp_db: Store, sample_skill: Skill) -> None:
    tmp_db.add_skill(sample_skill, embedding=[0.1, 0.2, 0.3, 0.4])
    tmp_db.delete_skill(sample_skill.id)
    assert tmp_db.get_skill(sample_skill.id) is None


def test_list_skills_ordering(tmp_db: Store) -> None:
    for i, score in enumerate([0.3, 0.9, 0.5]):
        s = Skill(
            activation=f"skill {i}",
            execution="step",
            termination="done",
            invocations=10,
            successes=int(score * 10),
        )
        s.recompute_score()
        tmp_db.add_skill(s, embedding=[0.1 * i, 0.0, 0.0, 0.0])

    ordered = tmp_db.list_skills()
    assert len(ordered) == 3
    assert ordered[0].score == 0.9
    assert ordered[-1].score == 0.3


def test_count_skills(tmp_db: Store) -> None:
    assert tmp_db.count_skills() == 0
    for i in range(5):
        tmp_db.add_skill(
            Skill(activation=f"s{i}", execution="e", termination="t"),
            embedding=[0.0, 0.0, 0.0, float(i)],
        )
    assert tmp_db.count_skills() == 5
    assert tmp_db.count_skills(scope=Scope.PROJECT) == 5
    assert tmp_db.count_skills(scope=Scope.GLOBAL) == 0


def test_list_live_skills_includes_legacy_established_rows(tmp_db: Store) -> None:
    skill = Skill(
        activation="legacy established skill",
        execution="do the thing",
        termination="done",
        maturity=Maturity.LIVE,
    )
    tmp_db.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])

    with tmp_db.batch() as conn:
        conn.execute("UPDATE skills SET maturity = 'established' WHERE id = ?", (skill.id,))

    live_skills = tmp_db.list_skills(maturity=Maturity.LIVE)
    assert any(item.id == skill.id for item in live_skills)


def test_search_by_embedding_returns_nearest(tmp_db: Store) -> None:
    # insert three skills with distinct embeddings
    embeddings = [
        ("close", [0.1, 0.1, 0.1, 0.1]),
        ("mid", [0.5, 0.5, 0.5, 0.5]),
        ("far", [1.0, 1.0, 1.0, 1.0]),
    ]
    for label, emb in embeddings:
        tmp_db.add_skill(
            Skill(activation=label, execution="e", termination="t"),
            embedding=emb,
        )

    # query close to first
    hits = tmp_db.search_skills_by_embedding([0.11, 0.11, 0.11, 0.11], top_k=3)
    assert len(hits) == 3
    # closest first
    assert hits[0][0].activation == "close"
    assert hits[0][1] < hits[1][1] < hits[2][1]


def test_embedding_dim_mismatch_raises(tmp_db: Store, sample_skill: Skill) -> None:
    import pytest

    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        tmp_db.add_skill(sample_skill, embedding=[0.1, 0.2])


def test_episode_round_trip(tmp_db: Store, successful_episode: Episode) -> None:
    tmp_db.add_episode(successful_episode)
    loaded = tmp_db.get_episode(successful_episode.id)
    assert loaded is not None
    assert loaded.outcome is Outcome.SUCCESS
    assert loaded.trajectory.num_tool_calls() == 3
    assert loaded.trajectory.tool_calls[0].error is not None
    assert loaded.trajectory.tool_calls[-1].result == "5 passed in 2.13s"
