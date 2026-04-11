"""Benchmark dataset builder and runner.

Builds a frozen JSON file of scored (skill, episode) activation pairs.
Run against it to detect regressions in retrieval or skill quality.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from muscle_memory.db import Store
from muscle_memory.embeddings import Embedder
from muscle_memory.eval.scorers import (
    load_activation_distances,
    score_adherence,
    score_correctness,
    score_relevance,
)
from muscle_memory.models import Episode


@dataclass
class BenchmarkEntry:
    skill_id: str
    skill_activation: str
    episode_id: str
    user_prompt: str
    relevance_score: float
    adherence_score: float
    correctness_verdict: str  # "correct" | "incorrect" | "needs_review"
    correctness_confidence: str  # "auto" | "human"
    outcome: str
    scored_at: str = ""


@dataclass
class BenchmarkRunResult:
    total: int = 0
    avg_relevance: float = 0.0
    avg_adherence: float = 0.0
    baseline_avg_relevance: float = 0.0
    baseline_avg_adherence: float = 0.0
    improved: list[dict[str, str]] = field(default_factory=list)
    degraded: list[dict[str, str]] = field(default_factory=list)


def build_benchmark(
    store: Store,
    *,
    embedder: Embedder | None = None,
    output_path: Path | None = None,
) -> tuple[list[BenchmarkEntry], Path]:
    """Score all (episode, skill) activation pairs and export as frozen JSON.

    Returns the entries and the path to the benchmark file.
    """
    episodes = store.list_episodes(limit=10_000)
    entries: list[BenchmarkEntry] = []
    now = datetime.now(UTC).isoformat()

    # Deduplicate by session: keep only latest episode per session
    by_session: dict[str, Episode] = {}
    for ep in episodes:
        sid = ep.session_id or ep.id
        existing = by_session.get(sid)
        if existing is None or ep.trajectory.num_tool_calls() > existing.trajectory.num_tool_calls():
            by_session[sid] = ep

    for ep in by_session.values():
        if not ep.activated_skills:
            continue

        # Load stored distances from sidecar
        distances = load_activation_distances(store.db_path, ep.session_id or "")

        for skill_id in set(ep.activated_skills):
            skill = store.get_skill(skill_id)
            if skill is None:
                continue

            # Score relevance
            rel = score_relevance(
                store, ep, skill_id,
                stored_distance=distances.get(skill_id),
                embedder=embedder,
            )

            # Score adherence
            adh = score_adherence(skill, ep.trajectory)

            # Score correctness
            cor = score_correctness(adh, ep.outcome)

            entries.append(
                BenchmarkEntry(
                    skill_id=skill_id,
                    skill_activation=skill.activation[:100],
                    episode_id=ep.id,
                    user_prompt=ep.user_prompt[:100],
                    relevance_score=rel.score,
                    adherence_score=adh.score,
                    correctness_verdict=cor.verdict,
                    correctness_confidence=cor.confidence,
                    outcome=ep.outcome.value,
                    scored_at=now,
                )
            )

    # Export
    if output_path is None:
        output_path = store.db_path.parent / "benchmark.json"

    data = {
        "version": 1,
        "created_at": now,
        "total_entries": len(entries),
        "entries": [asdict(e) for e in entries],
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n")

    return entries, output_path


def run_benchmark(
    store: Store,
    benchmark_path: Path,
    *,
    embedder: Embedder | None = None,
) -> BenchmarkRunResult:
    """Re-score against a frozen benchmark and report diffs."""
    raw = json.loads(benchmark_path.read_text())
    baseline_entries = [BenchmarkEntry(**e) for e in raw["entries"]]

    if not baseline_entries:
        return BenchmarkRunResult()

    improved: list[dict[str, str]] = []
    degraded: list[dict[str, str]] = []
    new_relevances: list[float] = []
    new_adherences: list[float] = []
    baseline_relevances: list[float] = []
    baseline_adherences: list[float] = []

    for entry in baseline_entries:
        ep = store.get_episode(entry.episode_id)
        skill = store.get_skill(entry.skill_id)
        if ep is None or skill is None:
            continue

        # Re-score with current code/data
        rel = score_relevance(store, ep, entry.skill_id, embedder=embedder)
        adh = score_adherence(skill, ep.trajectory)

        new_relevances.append(rel.score)
        new_adherences.append(adh.score)
        baseline_relevances.append(entry.relevance_score)
        baseline_adherences.append(entry.adherence_score)

        rel_delta = rel.score - entry.relevance_score
        adh_delta = adh.score - entry.adherence_score

        if rel_delta > 0.05 or adh_delta > 0.05:
            improved.append(
                {
                    "skill_id": entry.skill_id[:8],
                    "activation": entry.skill_activation[:50],
                    "relevance": f"{entry.relevance_score:.2f} -> {rel.score:.2f}",
                    "adherence": f"{entry.adherence_score:.2f} -> {adh.score:.2f}",
                }
            )
        elif rel_delta < -0.05 or adh_delta < -0.05:
            degraded.append(
                {
                    "skill_id": entry.skill_id[:8],
                    "activation": entry.skill_activation[:50],
                    "relevance": f"{entry.relevance_score:.2f} -> {rel.score:.2f}",
                    "adherence": f"{entry.adherence_score:.2f} -> {adh.score:.2f}",
                }
            )

    n = len(new_relevances) or 1
    return BenchmarkRunResult(
        total=n,
        avg_relevance=sum(new_relevances) / n,
        avg_adherence=sum(new_adherences) / n,
        baseline_avg_relevance=sum(baseline_relevances) / n,
        baseline_avg_adherence=sum(baseline_adherences) / n,
        improved=improved,
        degraded=degraded,
    )
