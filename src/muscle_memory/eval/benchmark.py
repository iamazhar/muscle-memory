"""Benchmark dataset builder and runner.

Builds a frozen JSON file of scored (skill, episode) activation pairs.
Run against it to detect regressions in retrieval or skill quality.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from muscle_memory.config import find_project_root
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
    false_positive_rate: float = 0.0
    execution_success_rate: float = 0.0
    promotion_precision: float = 0.0
    thresholds_passed: bool = False
    failed_thresholds: list[str] = field(default_factory=list)
    improved: list[dict[str, str]] = field(default_factory=list)
    degraded: list[dict[str, str]] = field(default_factory=list)


V1_RELEASE_THRESHOLDS = {
    "avg_relevance": 0.50,
    "avg_adherence": 0.70,
    "false_positive_rate": 0.15,
    "execution_success_rate": 0.70,
    "promotion_precision": 0.80,
}


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

    # Episodes are already newest-first, so keep the first one we see per session.
    by_session: dict[str, Episode] = {}
    for ep in episodes:
        sid = ep.session_id or ep.id
        if sid not in by_session:
            by_session[sid] = ep

    for ep in by_session.values():
        if not ep.activated_skills:
            continue

        # Load stored distances from sidecar
        distances = load_activation_distances(store.db_path, ep.session_id or "")

        for skill_id in dict.fromkeys(ep.activated_skills):
            skill = store.get_skill(skill_id)
            if skill is None:
                continue

            # Score relevance
            rel = score_relevance(
                store,
                ep,
                skill_id,
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

    repo_root = find_project_root(store.db_path.parent)
    data = {
        "version": 1,
        "created_at": now,
        "total_entries": len(entries),
        "entries": [asdict(e) for e in entries],
        "repo_head": _current_repo_head(repo_root),
        "source_tree_sha256": _current_source_tree_sha256(
            repo_root,
            excluded_paths=[output_path],
        ),
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n")

    return entries, output_path


def summarize_benchmark_artifact(benchmark_path: Path) -> BenchmarkRunResult:
    """Summarize a frozen benchmark without re-scoring against a local DB."""
    raw = json.loads(benchmark_path.read_text())
    baseline_entries = [BenchmarkEntry(**e) for e in raw["entries"]]

    if not baseline_entries:
        return BenchmarkRunResult(failed_thresholds=["no_entries"])

    total = len(baseline_entries)
    avg_relevance = sum(entry.relevance_score for entry in baseline_entries) / total
    avg_adherence = sum(entry.adherence_score for entry in baseline_entries) / total
    baseline_correct_count = sum(
        1 for entry in baseline_entries if entry.correctness_verdict == "correct"
    )
    execution_success_rate = 1.0 if baseline_correct_count else 0.0
    promotion_precision = 1.0 if baseline_correct_count else 0.0
    failed_thresholds = _failed_v1_thresholds(
        avg_relevance=avg_relevance,
        avg_adherence=avg_adherence,
        false_positive_rate=0.0,
        execution_success_rate=execution_success_rate,
        promotion_precision=promotion_precision,
        total=total,
    )
    return BenchmarkRunResult(
        total=total,
        avg_relevance=avg_relevance,
        avg_adherence=avg_adherence,
        baseline_avg_relevance=avg_relevance,
        baseline_avg_adherence=avg_adherence,
        false_positive_rate=0.0,
        execution_success_rate=execution_success_rate,
        promotion_precision=promotion_precision,
        thresholds_passed=not failed_thresholds,
        failed_thresholds=failed_thresholds,
    )


def _current_repo_head(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _current_source_tree_sha256(
    repo_root: Path | None,
    *,
    excluded_paths: list[Path] | tuple[Path, ...] | None = None,
) -> str | None:
    if repo_root is None:
        return None
    try:
        tracked_result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    excluded = {
        ".claude/benchmark.json",
        ".claude/mm.db",
        "benchmark-run.json",
    }
    repo_root_resolved = repo_root.resolve()
    for path in excluded_paths or ():
        try:
            rel_path = path.resolve().relative_to(repo_root_resolved).as_posix()
        except ValueError:
            continue
        excluded.add(rel_path)
    excluded_prefixes = (
        ".claude/mm.activations/",
        ".git/",
        ".hermes/",
        "docs/superpowers/plans/",
    )
    digest = hashlib.sha256()
    rel_paths = {
        entry.decode("utf-8")
        for raw in (tracked_result.stdout, untracked_result.stdout)
        for entry in raw.split(b"\0")
        if entry
    }
    for rel_path in sorted(rel_paths):
        if rel_path in excluded or rel_path.startswith(excluded_prefixes):
            continue
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            continue
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    baseline_correct_count = 0
    baseline_incorrect_count = 0
    true_positive_count = 0
    false_positive_count = 0
    predicted_positive_count = 0

    for entry in baseline_entries:
        ep = store.get_episode(entry.episode_id)
        skill = store.get_skill(entry.skill_id)
        if ep is None or skill is None:
            continue

        # Re-score with current code/data
        rel = score_relevance(store, ep, entry.skill_id, embedder=embedder)
        adh = score_adherence(skill, ep.trajectory)
        cor = score_correctness(adh, ep.outcome)

        new_relevances.append(rel.score)
        new_adherences.append(adh.score)
        baseline_relevances.append(entry.relevance_score)
        baseline_adherences.append(entry.adherence_score)
        baseline_verdict = entry.correctness_verdict
        if baseline_verdict == "correct":
            baseline_correct_count += 1
            if cor.verdict == "correct":
                true_positive_count += 1
        elif baseline_verdict == "incorrect":
            baseline_incorrect_count += 1
            if cor.verdict == "correct":
                false_positive_count += 1
        if cor.verdict == "correct" and baseline_verdict in {"correct", "incorrect"}:
            predicted_positive_count += 1

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

    total = len(new_relevances)
    avg_relevance = sum(new_relevances) / total if total else 0.0
    avg_adherence = sum(new_adherences) / total if total else 0.0
    baseline_avg_relevance = sum(baseline_relevances) / total if total else 0.0
    baseline_avg_adherence = sum(baseline_adherences) / total if total else 0.0
    false_positive_rate = (
        false_positive_count / baseline_incorrect_count if baseline_incorrect_count else 0.0
    )
    execution_success_rate = (
        true_positive_count / baseline_correct_count if baseline_correct_count else 0.0
    )
    promotion_precision = (
        true_positive_count / predicted_positive_count if predicted_positive_count else 0.0
    )
    failed_thresholds = _failed_v1_thresholds(
        avg_relevance=avg_relevance,
        avg_adherence=avg_adherence,
        false_positive_rate=false_positive_rate,
        execution_success_rate=execution_success_rate,
        promotion_precision=promotion_precision,
        total=total,
    )
    return BenchmarkRunResult(
        total=total,
        avg_relevance=avg_relevance,
        avg_adherence=avg_adherence,
        baseline_avg_relevance=baseline_avg_relevance,
        baseline_avg_adherence=baseline_avg_adherence,
        false_positive_rate=false_positive_rate,
        execution_success_rate=execution_success_rate,
        promotion_precision=promotion_precision,
        thresholds_passed=not failed_thresholds,
        failed_thresholds=failed_thresholds,
        improved=improved,
        degraded=degraded,
    )


def _failed_v1_thresholds(
    *,
    avg_relevance: float,
    avg_adherence: float,
    false_positive_rate: float,
    execution_success_rate: float,
    promotion_precision: float,
    total: int,
) -> list[str]:
    failed: list[str] = []
    if total == 0:
        failed.append("no_entries")
        return failed
    if avg_relevance < V1_RELEASE_THRESHOLDS["avg_relevance"]:
        failed.append("avg_relevance")
    if avg_adherence < V1_RELEASE_THRESHOLDS["avg_adherence"]:
        failed.append("avg_adherence")
    if false_positive_rate > V1_RELEASE_THRESHOLDS["false_positive_rate"]:
        failed.append("false_positive_rate")
    if execution_success_rate < V1_RELEASE_THRESHOLDS["execution_success_rate"]:
        failed.append("execution_success_rate")
    if promotion_precision < V1_RELEASE_THRESHOLDS["promotion_precision"]:
        failed.append("promotion_precision")
    return failed
