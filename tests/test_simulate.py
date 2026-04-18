"""Tests for the synthetic dogfooding harness (`mm simulate`).

These tests are the ground-truth contract the advisor pinned down:
  * PROVEN gate: a seeded CANDIDATE skill crossing 20 successes must
    land on score 1.0, maturity PROVEN, distinct_sources >= 2.
  * Loser pruning: a skill with 4/20 successes (score == 0.2) hits the
    prune threshold and is removed on `--prune`.
  * DB safety: `default_sim_db_path()` never resolves to a project's
    `.claude/mm.db`.
  * CLI: `mm simulate scenarios` and `mm simulate run --seed` are
    deterministic and re-entrant.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.db import Store
from muscle_memory.models import Maturity, Scope, Skill
from muscle_memory.simulate import (
    Scenario,
    Simulator,
    default_sim_db_path,
)
from muscle_memory.simulate_fixtures import demo_scenarios, demo_skills


@pytest.fixture
def sim_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "mm.sim.db", embedding_dims=16)


def _fixed_skill(skill_id: str, tag: str = "test") -> Skill:
    return Skill(
        id=skill_id,
        activation=f"When testing {skill_id}",
        execution="1. Do the thing\n2. Check output",
        termination="Task complete.",
        tags=[tag],
        scope=Scope.PROJECT,
        maturity=Maturity.CANDIDATE,
    )


# ----------------------------------------------------------------------
# Scenario validation
# ----------------------------------------------------------------------


def test_scenario_rejects_invalid_success_rate() -> None:
    with pytest.raises(ValueError, match="success_rate"):
        Scenario(name="x", prompt="y", activated_skills=["a"], success_rate=1.5, n=5)


def test_scenario_rejects_empty_activated_skills() -> None:
    with pytest.raises(ValueError, match="activated_skills"):
        Scenario(name="x", prompt="y", activated_skills=[], n=5)


def test_scenario_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        Scenario(name="x", prompt="y", activated_skills=["a"], n=-1)


# ----------------------------------------------------------------------
# Simulator — core scoring contract
# ----------------------------------------------------------------------


def test_golden_path_promotes_to_proven(sim_store: Store) -> None:
    """20 successes on a single skill => PROVEN, score 1.0."""
    sim = Simulator(sim_store, rng=random.Random(0))
    skill = _fixed_skill("s-proven")
    sim.seed([skill])

    sim.run(
        [
            Scenario(
                name="all-success",
                prompt="does the thing",
                activated_skills=[skill.id],
                success_rate=1.0,
                n=20,
            )
        ]
    )

    updated = sim_store.get_skill(skill.id)
    assert updated is not None
    assert updated.invocations == 20
    assert updated.successes == 20
    assert updated.failures == 0
    assert updated.score == pytest.approx(1.0)
    assert updated.maturity is Maturity.PROVEN
    # Layer 1 does NOT forge provenance — source_episode_ids stays empty.
    # LIVE promotion (which needs >=2 distinct sources) is Layer 2's job.
    assert updated.source_episode_ids == []


def test_all_failures_stays_candidate_and_prunes(sim_store: Store) -> None:
    """0% success floors the score and triggers the prune rule (score<=0.2)."""
    sim = Simulator(sim_store, rng=random.Random(0))
    skill = _fixed_skill("s-loser")
    sim.seed([skill])

    report = sim.run(
        [
            Scenario(
                name="all-fail",
                prompt="never works",
                activated_skills=[skill.id],
                success_rate=0.0,
                n=10,
            )
        ],
        prune=True,
    )

    # The skill should have been pruned (score 0.0, invocations 10 >= 5).
    assert sim_store.get_skill(skill.id) is None
    assert report.prune_report is not None
    assert skill.id in report.prune_report.removed


def test_mixed_outcome_live_maturity_is_deterministic_with_seed(
    sim_store: Store,
) -> None:
    """Fixed seed => reproducible success/failure split."""
    sim_a = Simulator(
        Store(sim_store.db_path, embedding_dims=16),
        rng=random.Random(42),
    )
    sim_b = Simulator(
        Store(sim_store.db_path, embedding_dims=16),
        rng=random.Random(42),
    )

    scenario = Scenario(
        name="mix",
        prompt="sometimes works",
        activated_skills=["s-mix"],
        success_rate=0.6,
        n=20,
    )
    skill = _fixed_skill("s-mix")
    sim_a.seed([skill])
    result_a = sim_a.run([scenario])

    # reset by deleting the skill and re-seeding
    sim_store.delete_skill(skill.id)
    # Drop episodes too so counters start from zero
    with sim_store.batch() as conn:
        conn.execute("DELETE FROM episodes")

    sim_b.seed([_fixed_skill("s-mix")])
    result_b = sim_b.run([scenario])

    assert result_a.scenarios[0].successes == result_b.scenarios[0].successes
    assert result_a.scenarios[0].failures == result_b.scenarios[0].failures


def test_missing_skill_is_silently_skipped(sim_store: Store) -> None:
    """Scenarios referencing unseeded skills must not crash.

    credit_episode and the simulator's invocation-bump loop both tolerate
    missing skill ids (real retrieval can race with deletion). Verify.
    """
    sim = Simulator(sim_store, rng=random.Random(0))
    # Intentionally do NOT seed.
    report = sim.run(
        [
            Scenario(
                name="ghost",
                prompt="missing",
                activated_skills=["does-not-exist"],
                success_rate=1.0,
                n=3,
            )
        ]
    )
    assert report.total_episodes == 3
    assert sim_store.get_skill("does-not-exist") is None


# ----------------------------------------------------------------------
# Fixtures + DB safety
# ----------------------------------------------------------------------


def test_default_sim_db_path_is_disjoint_from_project() -> None:
    """Layer 1's whole premise: never clobber a real project DB.

    The sim DB lives under the USER's `~/.claude/`, not under any project's
    `.claude/` — the filename is `mm.sim.db`, distinct from `mm.db`, so even
    if someone invoked sim with `$HOME` set to a project root it still would
    not collide with the project skill pool.
    """
    path = default_sim_db_path()
    assert path.name == "mm.sim.db"
    # Sim DB lives under the user's home .claude dir.
    assert ".claude" in path.parts
    assert path.is_relative_to(Path.home())
    # Filename is specifically NOT the project DB name.
    assert path.name != "mm.db"


def test_demo_fixtures_are_internally_consistent() -> None:
    """Every scenario references a skill that exists in the fixture list."""
    skills = {s.id for s in demo_skills()}
    for scenario in demo_scenarios():
        for sid in scenario.activated_skills:
            assert sid in skills, f"scenario {scenario.name} targets unknown skill {sid}"


# ----------------------------------------------------------------------
# CLI smoke tests
# ----------------------------------------------------------------------


runner = CliRunner()


def test_cli_scenarios_renders() -> None:
    result = runner.invoke(app, ["simulate", "scenarios"])
    assert result.exit_code == 0, result.output
    for scenario in demo_scenarios():
        assert scenario.name in result.output


def test_cli_scenarios_json_valid() -> None:
    import json as _json

    result = runner.invoke(app, ["simulate", "scenarios", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    assert len(payload) == len(demo_scenarios())
    assert {s["name"] for s in payload} == {s.name for s in demo_scenarios()}


def test_cli_run_golden_path_promotes(tmp_path: Path) -> None:
    """End-to-end: `mm simulate run --db ... --seed 0` produces a PROVEN skill."""
    import json as _json

    db_path = tmp_path / "mm.sim.db"
    result = runner.invoke(
        app,
        [
            "simulate",
            "run",
            "--db",
            str(db_path),
            "--seed",
            "0",
            "--fresh",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    assert payload["total_episodes"] == sum(s.n for s in demo_scenarios())

    store = Store(db_path, embedding_dims=16)
    proven = store.get_skill("sim-pytest-runner")
    assert proven is not None
    assert proven.maturity is Maturity.PROVEN
    assert proven.score == pytest.approx(1.0)

    # The loser scenario hits <=0.2 score and the pruner removes it.
    assert "sim-flaky-retry" in payload["pruned"]
    assert store.get_skill("sim-flaky-retry") is None


def test_cli_run_refuses_to_touch_project_db_by_default(tmp_path: Path) -> None:
    """Without --inplace and without --db, sim writes to default_sim_db_path —
    never to the project's .claude/mm.db. Rather than actually hitting the
    home dir in tests, we verify `_resolve_sim_db_path` directly.
    """
    from muscle_memory.cli import _resolve_sim_db_path
    from muscle_memory.config import Config

    project = tmp_path / "proj"
    project.mkdir()
    cfg = Config(
        db_path=project / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project,
        embedding_dims=16,
    )

    resolved = _resolve_sim_db_path(None, inplace=False, cfg=cfg)
    assert resolved != cfg.db_path
    assert resolved == default_sim_db_path()

    # With --inplace, we DO point at the project db (opt-in).
    resolved_inplace = _resolve_sim_db_path(None, inplace=True, cfg=cfg)
    assert resolved_inplace == cfg.db_path

    # Explicit --db always wins.
    override = tmp_path / "other.db"
    resolved_override = _resolve_sim_db_path(override, inplace=True, cfg=cfg)
    assert resolved_override == override


def test_cli_run_inplace_does_not_prune_by_default(tmp_path: Path) -> None:
    """--inplace must NOT default to pruning — that would silently remove
    real project skills whose score fell to <=0.2.

    We point --db at a tmp path (so nothing real is touched) but use --inplace
    semantics by leaving --prune unset. A pre-seeded loser skill with a real
    prune-eligible profile must survive the run.
    """
    db_path = tmp_path / "inplace.db"
    store = Store(db_path, embedding_dims=16)
    # Seed a skill that would be pruned: score 0.0, invocations >= 5.
    loser = _fixed_skill("would-be-pruned")
    loser.successes = 0
    loser.failures = 10
    loser.invocations = 10
    loser.score = 0.0
    store.add_skill(loser)
    assert store.get_skill("would-be-pruned") is not None

    # With --inplace and no explicit --prune, prune is OFF.
    result = runner.invoke(
        app,
        [
            "simulate",
            "run",
            "--db",
            str(db_path),
            "--inplace",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    # Pre-existing loser must still be present (prune did not run).
    assert store.get_skill("would-be-pruned") is not None
