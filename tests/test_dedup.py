"""Tests for skill deduplication."""

from __future__ import annotations

from collections.abc import Iterable

from muscle_memory.db import Store
from muscle_memory.dedup import (
    add_skill_with_dedup,
    consolidate_group,
    find_near_duplicate_groups,
)
from muscle_memory.models import Maturity, Skill


class FakeEmbedder:
    """4-dim deterministic embedder for dedup tests.

    Returns a vector shaped by the text content so we can engineer
    precise distances between embeddings.
    """

    dims = 4

    def __init__(self, overrides: dict[str, list[float]] | None = None):
        self._overrides = overrides or {}

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        if text in self._overrides:
            return list(self._overrides[text])
        # default: bag of char codes mod 4
        v = [0.0, 0.0, 0.0, 0.0]
        for c in text.lower():
            v[ord(c) % 4] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


def _make_skill(activation: str, **kw) -> Skill:
    defaults = {
        "execution": "do the thing",
        "termination": "thing is done",
    }
    defaults.update(kw)
    return Skill(activation=activation, **defaults)


def test_add_skill_with_dedup_inserts_when_empty(tmp_db: Store) -> None:
    embedder = FakeEmbedder({"new skill": [0.1, 0.2, 0.3, 0.4]})
    skill = _make_skill("new skill")
    added, existing = add_skill_with_dedup(tmp_db, embedder, skill)
    assert added is True
    assert existing is None
    assert tmp_db.count_skills() == 1


def test_add_skill_with_dedup_rejects_close_neighbor(tmp_db: Store) -> None:
    embedder = FakeEmbedder(
        {
            "old skill": [0.1, 0.2, 0.3, 0.4],
            "near dup": [0.11, 0.21, 0.31, 0.41],
        }
    )
    first = _make_skill("old skill", successes=5, invocations=6)
    first.recompute_score()
    tmp_db.add_skill(first, embedding=embedder.embed_one("old skill"))

    second = _make_skill("near dup")
    added, existing = add_skill_with_dedup(tmp_db, embedder, second)
    assert added is False
    assert existing is not None
    assert existing.id == first.id
    assert tmp_db.count_skills() == 1


def test_add_skill_with_dedup_allows_distant_skill(tmp_db: Store) -> None:
    embedder = FakeEmbedder(
        {
            "old skill": [1.0, 0.0, 0.0, 0.0],
            "distant skill": [0.0, 1.0, 0.0, 0.0],
        }
    )
    tmp_db.add_skill(_make_skill("old skill"), embedding=embedder.embed_one("old skill"))
    added, _existing = add_skill_with_dedup(tmp_db, embedder, _make_skill("distant skill"))
    assert added is True
    assert tmp_db.count_skills() == 2


def test_consolidate_group_preserves_history(tmp_db: Store) -> None:
    """Merging should sum counts and union episode ids into the keeper."""
    keeper = _make_skill(
        "best skill",
        successes=5,
        invocations=8,
        failures=1,
        maturity=Maturity.ESTABLISHED,
        source_episode_ids=["ep1", "ep2"],
    )
    keeper.recompute_score()
    tmp_db.add_skill(keeper, embedding=[1.0, 0.0, 0.0, 0.0])

    loser1 = _make_skill(
        "second best",
        successes=2,
        invocations=3,
        failures=0,
        source_episode_ids=["ep3"],
    )
    loser1.recompute_score()
    tmp_db.add_skill(loser1, embedding=[1.0, 0.01, 0.0, 0.0])

    loser2 = _make_skill(
        "third",
        successes=1,
        invocations=1,
        source_episode_ids=["ep2", "ep4"],  # ep2 overlap
    )
    loser2.recompute_score()
    tmp_db.add_skill(loser2, embedding=[1.0, 0.02, 0.0, 0.0])

    merged = consolidate_group(tmp_db, [keeper, loser1, loser2])
    assert merged is not None
    assert merged.id == keeper.id
    assert merged.successes == 5 + 2 + 1
    assert merged.invocations == 8 + 3 + 1
    assert merged.failures == 1
    # ep2 overlap → no duplicate
    assert merged.source_episode_ids == ["ep1", "ep2", "ep3", "ep4"]

    assert tmp_db.get_skill(loser1.id) is None
    assert tmp_db.get_skill(loser2.id) is None

    reloaded = tmp_db.get_skill(keeper.id)
    assert reloaded is not None
    assert reloaded.successes == 8
    assert reloaded.invocations == 12


def test_find_near_duplicate_groups_orders_keeper_by_quality(
    tmp_db: Store,
) -> None:
    """The first skill in each group should be the 'best' one.

    The FakeEmbedder must produce the SAME vectors at store-time and
    at cluster-time, otherwise the store KNN query uses different
    embeddings than the grouping logic expects.
    """
    embedder = FakeEmbedder(
        {
            "var 1 of a thing": [1.0, 0.0, 0.0, 0.0],
            "var 2 of a thing": [1.0, 0.001, 0.0, 0.0],
            "var 3 of a thing": [1.0, 0.002, 0.0, 0.0],
        }
    )

    a = _make_skill("var 1 of a thing", successes=1, invocations=2)
    a.recompute_score()
    tmp_db.add_skill(a, embedding=embedder.embed_one(a.activation))

    b = _make_skill("var 2 of a thing", successes=5, invocations=5)
    b.recompute_score()
    tmp_db.add_skill(b, embedding=embedder.embed_one(b.activation))

    c = _make_skill("var 3 of a thing", successes=2, invocations=3)
    c.recompute_score()
    tmp_db.add_skill(c, embedding=embedder.embed_one(c.activation))

    groups = find_near_duplicate_groups(tmp_db, embedder, threshold=0.1)
    assert len(groups) == 1
    assert groups[0][0].id == b.id  # highest successes → keeper


def test_find_near_duplicate_groups_empty_on_singleton(tmp_db: Store) -> None:
    embedder = FakeEmbedder()
    s = _make_skill("only one")
    tmp_db.add_skill(s, embedding=[1.0, 0.0, 0.0, 0.0])
    groups = find_near_duplicate_groups(tmp_db, embedder)
    assert groups == []
