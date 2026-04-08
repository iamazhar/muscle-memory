"""Tests for the Retriever."""

from __future__ import annotations

from typing import Iterable

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Maturity, Skill
from muscle_memory.retriever import Retriever


class FakeEmbedder:
    """Deterministic embedder: hash text → 4-dim vector."""

    dims = 4

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        # Use word fingerprint as a crude deterministic vector.
        words = text.lower().split()
        v = [0.0, 0.0, 0.0, 0.0]
        for w in words:
            h = sum(ord(c) for c in w) % 4
            v[h] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


def test_retriever_finds_closest_skill(tmp_db: Store, sample_config: Config) -> None:
    embedder = FakeEmbedder()

    # two distinct skills
    a = Skill(
        activation="pytest fails with ModuleNotFoundError",
        execution="use tools/test-runner.sh",
        termination="tests pass",
    )
    b = Skill(
        activation="deploy to production requires approval",
        execution="ask in #deploys channel first",
        termination="approved",
    )
    tmp_db.add_skill(a, embedding=embedder.embed_one(a.activation))
    tmp_db.add_skill(b, embedding=embedder.embed_one(b.activation))

    retriever = Retriever(tmp_db, embedder, sample_config)
    hits = retriever.retrieve("pytest failing with import errors")
    assert hits, "expected at least one hit"
    assert hits[0].skill.id == a.id


def test_retriever_handles_empty_store(tmp_db: Store, sample_config: Config) -> None:
    retriever = Retriever(tmp_db, FakeEmbedder(), sample_config)
    assert retriever.retrieve("anything") == []


def test_retriever_skips_empty_query(tmp_db: Store, sample_config: Config) -> None:
    retriever = Retriever(tmp_db, FakeEmbedder(), sample_config)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_mark_activated_updates_invocations(
    tmp_db: Store, sample_config: Config
) -> None:
    embedder = FakeEmbedder()
    s = Skill(activation="pytest tests fail", execution="e", termination="t")
    tmp_db.add_skill(s, embedding=embedder.embed_one(s.activation))

    retriever = Retriever(tmp_db, embedder, sample_config)
    hits = retriever.retrieve("pytest is failing")
    assert hits
    before = hits[0].skill.invocations

    retriever.mark_activated(hits)
    reloaded = tmp_db.get_skill(s.id)
    assert reloaded is not None
    assert reloaded.invocations == before + 1
    assert reloaded.last_used_at is not None


def test_proven_skill_ranks_above_candidate_at_same_distance(
    tmp_db: Store, sample_config: Config
) -> None:
    embedder = FakeEmbedder()

    a = Skill(activation="deploy staging", execution="e", termination="t")
    b = Skill(
        activation="deploy staging",
        execution="e",
        termination="t",
        maturity=Maturity.PROVEN,
        invocations=20,
        successes=18,
        score=0.9,
    )
    tmp_db.add_skill(a, embedding=embedder.embed_one(a.activation))
    tmp_db.add_skill(b, embedding=embedder.embed_one(b.activation))

    retriever = Retriever(tmp_db, embedder, sample_config)
    hits = retriever.retrieve("deploy staging")
    assert hits
    # the proven one should come first despite identical distance
    assert hits[0].skill.maturity is Maturity.PROVEN
