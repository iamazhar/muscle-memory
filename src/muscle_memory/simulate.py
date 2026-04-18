"""Synthetic dogfooding harness (Layer 1).

Drives skills through the scoring / promotion / pruning pipeline without
real sessions, real LLM calls, or real retrieval. Answers the question
"does a seeded CANDIDATE skill cross the PROVEN gate after N successes?"
in milliseconds rather than the hours it takes to run real agent
sessions.

Design choices (advisor-reframed):
  * No extractor. Skills are hand-written fixtures, pre-seeded into the
    DB. We're validating the scoring seam, not LLM behavior.
  * No retriever. A `Scenario` declares which skill ids activate — the
    simulator just records them into `Episode.activated_skills` and lets
    the Scorer do its job.
  * No embeddings by default. Skills are stored without vectors, which
    means `mm retrieve` won't find them, but `mm list` / `mm stats` /
    `mm review` all work, and that's the whole point: observable end
    state for manual inspection and deterministic tests.

What Layer 1 *cannot* cover:
  * LIVE promotion requires ``>= 2 distinct source_episode_ids``
    (models.py:206). `source_episode_ids` only grows when the *extractor*
    re-extracts a skill from a new episode — pure usage (which is all
    Layer 1 simulates) never mutates it. So Layer 1 drives PROVEN
    promotion (>=10 successes + score >=0.75) and the loser-prune path,
    but LIVE transitions need Layer 2 (transcript replay through the
    extractor). This is by design — faking provenance would hide the
    real bottleneck in the promotion pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome, Skill, ToolCall, Trajectory
from muscle_memory.scorer import PruneReport, Scorer


@dataclass
class Scenario:
    """A synthetic workload unit.

    `n` episodes are generated. For each, `activated_skills` is credited
    with either SUCCESS (with probability `success_rate`) or FAILURE.
    """

    name: str
    prompt: str
    activated_skills: list[str]
    success_rate: float = 1.0
    n: int = 20
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError(f"success_rate must be in [0, 1]; got {self.success_rate}")
        if self.n < 0:
            raise ValueError(f"n must be >= 0; got {self.n}")
        if not self.activated_skills:
            raise ValueError("activated_skills must not be empty — the scorer needs a target")


@dataclass
class ScenarioResult:
    name: str
    successes: int
    failures: int
    episodes_written: int


@dataclass
class SimulationReport:
    scenarios: list[ScenarioResult]
    total_episodes: int
    prune_report: PruneReport | None = None

    def summary_line(self) -> str:
        return f"{self.total_episodes} episodes across {len(self.scenarios)} scenarios" + (
            f" — pruned {len(self.prune_report.removed)}, kept {self.prune_report.kept}"
            if self.prune_report is not None
            else ""
        )


def _synthetic_trajectory(prompt: str, outcome: Outcome, activated_skills: list[str]) -> Trajectory:
    """Build a minimal trajectory whose shape is consistent with outcome.

    The extractor never sees these (we skip that layer), and the scorer
    credits based on `episode.outcome`, not tool calls. We keep a single
    tool call so the trajectory isn't trivially rejected by any other
    consumer that spot-checks `num_tool_calls`.
    """
    ok = outcome is Outcome.SUCCESS
    return Trajectory(
        user_prompt=prompt,
        tool_calls=[
            ToolCall(
                name="Bash",
                arguments={"command": "echo simulated"},
                result="ok" if ok else None,
                error=None if ok else "simulated failure",
            )
        ],
        assistant_turns=[f"simulated run — activated={','.join(activated_skills)}"],
    )


class Simulator:
    """Runs scenarios against a Store, crediting skills via the Scorer."""

    def __init__(
        self,
        store: Store,
        *,
        rng: random.Random | None = None,
        max_skills: int = 500,
    ):
        self.store = store
        self.scorer = Scorer(store, max_skills=max_skills)
        self.rng = rng or random.Random()

    def seed(self, skills: list[Skill]) -> list[str]:
        """Pre-seed CANDIDATE skills into the store. Returns their ids."""
        ids: list[str] = []
        for skill in skills:
            if self.store.get_skill(skill.id) is None:
                # Stored without embeddings — retrieval is out of scope
                # for Layer 1; scoring/promotion/pruning are the target.
                self.store.add_skill(skill)
            ids.append(skill.id)
        return ids

    def run(self, scenarios: list[Scenario], *, prune: bool = False) -> SimulationReport:
        results: list[ScenarioResult] = []
        total = 0

        for scenario in scenarios:
            result = self._run_one(scenario)
            results.append(result)
            total += result.episodes_written

        prune_report = self.scorer.prune() if prune else None
        return SimulationReport(
            scenarios=results,
            total_episodes=total,
            prune_report=prune_report,
        )

    def _run_one(self, scenario: Scenario) -> ScenarioResult:
        successes = 0
        failures = 0
        now = datetime.now(UTC)

        # Bump invocations up-front — credit_episode only touches
        # successes/failures, mirroring real hook flow where retrieval
        # already bumped the invocation counter before the episode ran.
        for skill_id in scenario.activated_skills:
            skill = self.store.get_skill(skill_id)
            if skill is None:
                continue
            # pre-increment happens per-episode below

        for _i in range(scenario.n):
            is_success = self.rng.random() < scenario.success_rate
            outcome = Outcome.SUCCESS if is_success else Outcome.FAILURE

            # Bump invocations for each activated skill (retrieval-time
            # semantics). Only after that does credit_episode add the
            # success or failure delta. We deliberately do NOT append to
            # source_episode_ids here — that field is extractor-owned
            # provenance, and faking it would hide the real LIVE gate.
            for skill_id in scenario.activated_skills:
                skill = self.store.get_skill(skill_id)
                if skill is None:
                    continue
                skill.invocations += 1
                skill.last_used_at = now
                self.store.update_skill(skill)

            episode = Episode(
                user_prompt=scenario.prompt,
                trajectory=_synthetic_trajectory(
                    scenario.prompt, outcome, scenario.activated_skills
                ),
                outcome=outcome,
                reward=1.0 if is_success else -1.0,
                started_at=now,
                ended_at=now,
                activated_skills=list(scenario.activated_skills),
            )
            self.store.add_episode(episode)
            self.scorer.credit_episode(episode)

            if is_success:
                successes += 1
            else:
                failures += 1

        return ScenarioResult(
            name=scenario.name,
            successes=successes,
            failures=failures,
            episodes_written=scenario.n,
        )


def default_sim_db_path() -> Path:
    """Canonical location for the synthetic-dogfood DB.

    Deliberately *not* `.claude/mm.db` — we never want simulation to
    pollute the user's real skill pool. Stored under `~/.claude/` so
    it persists across runs for inspection but is disjoint from any
    project DB.
    """
    return Path.home() / ".claude" / "mm.sim.db"


__all__ = [
    "Scenario",
    "ScenarioResult",
    "SimulationReport",
    "Simulator",
    "default_sim_db_path",
]
