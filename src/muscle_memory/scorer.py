"""Credit assignment and skill pool maintenance.

When an Episode ends, we credit or discredit each Skill that was
active during it based on the inferred outcome. Scores and maturity
are recomputed, and low-value skills can be pruned.
"""

from __future__ import annotations

from dataclasses import dataclass

from muscle_memory.db import Store
from muscle_memory.models import Episode, Maturity, Outcome, Scope, Skill


@dataclass
class PruneReport:
    removed: list[str]
    kept: int


class Scorer:
    def __init__(self, store: Store, *, max_skills: int = 500):
        self.store = store
        self.max_skills = max_skills

    def credit_episode(self, episode: Episode) -> list[Skill]:
        """Update activated skills based on episode outcome.

        Returns the list of updated skills (for logging / UX).
        """
        if not episode.activated_skills:
            return []

        updated: list[Skill] = []
        for skill_id in episode.activated_skills:
            skill = self.store.get_skill(skill_id)
            if skill is None:
                continue

            if episode.outcome is Outcome.SUCCESS:
                skill.successes += 1
            elif episode.outcome is Outcome.FAILURE:
                skill.failures += 1
            # UNKNOWN: leave counts alone — we already bumped invocations at
            # retrieval time, so invocations > successes + failures is
            # "skill was used but outcome is unclear".

            skill.recompute_score()
            skill.recompute_maturity()
            self.store.update_skill(skill)
            updated.append(skill)
        return updated

    def prune(self, *, min_invocations_before_prune: int = 5) -> PruneReport:
        """Remove demonstrably bad skills and enforce pool capacity.

        Rules:
          1. A skill with >= `min_invocations_before_prune` invocations
             and score <= 0.2 is removed (it's been tried and it flops).
          2. If the pool still exceeds `max_skills`, remove the lowest-
             scoring CANDIDATE skills first, then ESTABLISHED, until
             under capacity.
        """
        removed: list[str] = []

        # rule 1: demonstrated losers
        for skill in self.store.list_skills():
            if (
                skill.invocations >= min_invocations_before_prune
                and skill.score <= 0.2
            ):
                self.store.delete_skill(skill.id)
                removed.append(skill.id)

        # rule 2: capacity
        while self.store.count_skills(scope=Scope.PROJECT) > self.max_skills:
            # fetch candidates cheapest first
            pool = self.store.list_skills(
                scope=Scope.PROJECT, maturity=Maturity.CANDIDATE
            )
            if not pool:
                pool = self.store.list_skills(
                    scope=Scope.PROJECT, maturity=Maturity.ESTABLISHED
                )
            if not pool:
                break
            pool.sort(key=lambda s: (s.score, s.invocations))
            victim = pool[0]
            self.store.delete_skill(victim.id)
            removed.append(victim.id)

        kept = self.store.count_skills()
        return PruneReport(removed=removed, kept=kept)
