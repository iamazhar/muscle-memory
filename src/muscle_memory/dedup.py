"""Skill deduplication — one place, used by every insertion path.

The extractor runs multiple times per session (once per turn-end in the
Stop hook's async pipeline). Without pre-insertion dedup, each run
generates micro-variants of the same skill and the store bloats fast.
This module is the shared gate every new skill must pass through.
"""

from __future__ import annotations

from muscle_memory.admission import candidate_ready_for_live
from muscle_memory.db import Store
from muscle_memory.embeddings import Embedder
from muscle_memory.models import Maturity, Skill

# Embedding distance below which two skills are considered duplicates.
# sqlite-vec returns L2 (euclidean) distance for FLOAT vectors.
# 0.0 = identical, sqrt(2)≈1.41 = orthogonal for normalized vectors.
#
# Empirically measured with BGE-small-en-v1.5 + Sonnet-authored skills,
# 5 micro-variants of the same "uv pth hidden flag" skill cluster with
# pairwise L2 distances in [0.25, 0.45]. The connected component under
# a 0.40 threshold catches all five as one group. Unrelated skills
# (e.g. different topics entirely) sit well above 0.60 in our data, so
# 0.40 leaves comfortable margin.
#
# Users can override via `mm dedup --threshold` for more/less aggression.
DEDUP_DISTANCE_THRESHOLD = 0.40


def add_skill_with_dedup(
    store: Store,
    embedder: Embedder,
    skill: Skill,
) -> tuple[bool, Skill | None]:
    """Insert `skill` unless a near-duplicate already exists.

    Returns:
        (True, None)    — skill inserted as new
        (False, existing) — skill skipped; `existing` is the dupe we
                            kept instead
    """
    embedding = embedder.embed_one(skill.activation)
    existing = find_duplicate(store, embedding)
    if existing is not None:
        changed = False

        seen_episodes = set(existing.source_episode_ids)
        for ep_id in skill.source_episode_ids:
            if ep_id not in seen_episodes:
                existing.source_episode_ids.append(ep_id)
                seen_episodes.add(ep_id)
                changed = True

        for hint in skill.tool_hints:
            if hint not in existing.tool_hints:
                existing.tool_hints.append(hint)
                changed = True
        for tag in skill.tags:
            if tag not in existing.tags:
                existing.tags.append(tag)
                changed = True

        if existing.maturity is Maturity.CANDIDATE and candidate_ready_for_live(existing):
            existing.maturity = Maturity.LIVE
            changed = True

        if changed:
            store.update_skill(existing)
        return False, existing

    store.add_skill(skill, embedding=embedding)
    return True, None


def find_duplicate(store: Store, embedding: list[float]) -> Skill | None:
    """Return the closest existing skill if it's within the dedup threshold."""
    hits = store.search_skills_by_embedding(embedding, top_k=1)
    if not hits:
        return None
    existing, distance = hits[0]
    if distance < DEDUP_DISTANCE_THRESHOLD:
        return existing
    return None


def find_near_duplicate_groups(
    store: Store,
    embedder: Embedder,
    threshold: float = DEDUP_DISTANCE_THRESHOLD,
) -> list[list[Skill]]:
    """Cluster existing skills into groups of near-duplicates.

    Uses embedding-based neighbor lookups to build adjacency, then
    extracts connected components. Each returned group has 2+ skills;
    isolated skills are omitted. Within each group, skills are ordered
    by quality descending (successes, then score, then invocations),
    so the first element is the natural "keeper".
    """
    skills = store.list_skills()
    if len(skills) < 2:
        return []

    embeddings = embedder.embed([s.activation for s in skills])
    skill_by_id = {s.id: s for s in skills}

    adjacency: dict[str, set[str]] = {s.id: set() for s in skills}
    for skill, emb in zip(skills, embeddings):
        hits = store.search_skills_by_embedding(emb, top_k=5)
        for other, distance in hits:
            if other.id == skill.id:
                continue
            if distance < threshold:
                adjacency[skill.id].add(other.id)
                adjacency[other.id].add(skill.id)

    visited: set[str] = set()
    groups: list[list[Skill]] = []
    for sid in adjacency:
        if sid in visited:
            continue
        component: list[str] = []
        stack = [sid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            stack.extend(adjacency[cur] - visited)
        if len(component) >= 2:
            members = [skill_by_id[c] for c in component]
            members.sort(
                key=lambda s: (s.successes, s.score, s.invocations, s.created_at),
                reverse=True,
            )
            groups.append(members)

    return groups


def consolidate_group(store: Store, group: list[Skill]) -> Skill | None:
    """Collapse a group of duplicate skills into the first (best) one.

    Merges successes/failures/invocations into the keeper, unions the
    source_episode_ids, recomputes score/maturity, and deletes the rest.
    Returns the surviving skill.
    """
    if not group:
        return None
    if len(group) == 1:
        return group[0]

    keeper = group[0]
    losers = group[1:]

    for loser in losers:
        keeper.invocations += loser.invocations
        keeper.successes += loser.successes
        keeper.failures += loser.failures
        seen = set(keeper.source_episode_ids)
        for ep_id in loser.source_episode_ids:
            if ep_id not in seen:
                keeper.source_episode_ids.append(ep_id)
                seen.add(ep_id)
        if loser.last_used_at and (
            keeper.last_used_at is None or loser.last_used_at > keeper.last_used_at
        ):
            keeper.last_used_at = loser.last_used_at

    keeper.recompute_score()
    keeper.recompute_maturity()
    store.update_skill(keeper)

    for loser in losers:
        store.delete_skill(loser.id)

    return keeper
