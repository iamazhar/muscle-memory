"""Automated scorers for playbook evaluation.

Three dimensions:
1. Relevance — did the playbook fire for the right situation?
2. Adherence — were the playbook's steps actually executed?
3. Correctness — did following the steps solve the problem?
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import json
from pathlib import Path

from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome, Skill, Trajectory


# ------------------------------------------------------------------
# Relevance: embedding similarity between prompt and skill activation
# ------------------------------------------------------------------


@dataclass
class RelevanceScore:
    score: float  # 0.0-1.0, higher = more relevant
    l2_distance: float  # raw L2 distance
    method: str  # "stored" | "recomputed"


def score_relevance(
    store: Store,
    episode: Episode,
    skill_id: str,
    *,
    stored_distance: float | None = None,
    embedder: object | None = None,
) -> RelevanceScore:
    """Score how relevant a skill was to the user's prompt.

    Uses stored L2 distance if available, otherwise recomputes from
    stored embeddings.
    """
    if stored_distance is not None:
        return RelevanceScore(
            score=_l2_to_similarity(stored_distance),
            l2_distance=stored_distance,
            method="stored",
        )

    # Recompute from stored embeddings
    skill_emb = store.get_skill_embedding(skill_id)
    if skill_emb is None or not episode.user_prompt:
        return RelevanceScore(score=0.0, l2_distance=2.0, method="recomputed")

    if embedder is None:
        # Can't recompute without an embedder
        return RelevanceScore(score=0.0, l2_distance=2.0, method="recomputed")

    # Embed the user prompt and compute L2 distance
    prompt_emb = embedder.embed_one(episode.user_prompt)  # type: ignore[union-attr]
    distance = _l2_distance(prompt_emb, skill_emb)
    return RelevanceScore(
        score=_l2_to_similarity(distance),
        l2_distance=distance,
        method="recomputed",
    )


def load_activation_distances(db_path: Path, session_id: str) -> dict[str, float]:
    """Load stored retrieval distances from the sidecar file.

    Returns a dict mapping skill_id -> L2 distance.
    Works with both old format (list of strings) and new format (list of dicts).
    """
    sidecar = db_path.parent / "mm.activations" / f"{session_id}.json"
    if not sidecar.exists():
        return {}
    try:
        raw = json.loads(sidecar.read_text())
        distances: dict[str, float] = {}
        for entry in raw:
            if isinstance(entry, dict) and entry.get("distance") is not None:
                distances[entry["skill_id"]] = entry["distance"]
        return distances
    except Exception:
        return {}


def _l2_to_similarity(distance: float) -> float:
    """Convert L2 distance to 0-1 similarity score.

    For normalized vectors (BGE-small-en-v1.5), L2 ranges from
    0 (identical) to ~2.0 (orthogonal).
    """
    return max(0.0, 1.0 - distance / 2.0)


def _l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ------------------------------------------------------------------
# Adherence: were the playbook's steps actually executed?
# ------------------------------------------------------------------


@dataclass
class AdherenceScore:
    score: float  # 0.0-1.0, fraction of steps matched
    matched_steps: list[str] = field(default_factory=list)
    unmatched_steps: list[str] = field(default_factory=list)
    total_steps: int = 0


def score_adherence(skill: Skill, trajectory: Trajectory) -> AdherenceScore:
    """Score whether the skill's execution steps appear in the trajectory."""
    steps = _parse_execution_steps(skill.execution)
    if not steps:
        return AdherenceScore(score=1.0, total_steps=0)

    # Build a searchable corpus from the trajectory
    corpus = _build_trajectory_corpus(trajectory)

    matched = []
    unmatched = []
    for step in steps:
        tokens = _extract_step_tokens(step)
        if _any_token_in_corpus(tokens, corpus):
            matched.append(step)
        else:
            unmatched.append(step)

    score = len(matched) / len(steps) if steps else 1.0
    return AdherenceScore(
        score=score,
        matched_steps=matched,
        unmatched_steps=unmatched,
        total_steps=len(steps),
    )


def _parse_execution_steps(execution: str) -> list[str]:
    """Split execution text into individual steps."""
    lines = execution.strip().split("\n")
    steps = []
    for line in lines:
        # Strip numbering: "1. Do X" or "1) Do X" or "- Do X"
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]\s*|-\s*|\*\s*)", "", line).strip()
        if cleaned and len(cleaned) > 3:
            steps.append(cleaned)
    return steps


def _extract_step_tokens(step: str) -> list[str]:
    """Extract matchable tokens from a step description.

    Looks for backtick-wrapped commands, file paths, and command names.
    """
    tokens = []

    # Backtick-wrapped commands: `pytest`, `git commit`, `chflags nohidden`
    backtick_matches = re.findall(r"`([^`]+)`", step)
    for match in backtick_matches:
        # Split compound commands to get individual tokens
        for part in match.split("&&"):
            part = part.strip()
            if part:
                # Take the first word as the command name
                cmd = part.split()[0] if part.split() else part
                tokens.append(cmd)
                # Also add the full command for substring matching
                tokens.append(part)

    # File paths: words containing / or ending in common extensions
    path_matches = re.findall(r"[\w./\-]+\.(?:py|js|ts|rb|rs|go|yaml|yml|json|toml|md|sh)", step)
    tokens.extend(path_matches)

    # Tool names: Bash, Edit, Write, Read, Grep, Glob
    tool_names = re.findall(r"\b(Bash|Edit|Write|Read|Grep|Glob|Agent)\b", step)
    tokens.extend(tool_names)

    # If no structured tokens found, use significant words from the step
    if not tokens:
        words = re.findall(r"\b[a-zA-Z_][\w\-]{3,}\b", step)
        # Filter out common English words
        stopwords = {
            "that", "this", "with", "from", "have", "been", "will",
            "should", "would", "could", "when", "then", "than", "each",
            "make", "sure", "also", "into", "some", "them", "their",
            "does", "done", "only", "just", "more", "most", "after",
            "before", "above", "below", "file", "step", "command",
        }
        tokens = [w for w in words if w.lower() not in stopwords]

    return tokens


def _build_trajectory_corpus(trajectory: Trajectory) -> str:
    """Build a searchable text corpus from all tool calls."""
    parts = []
    for tc in trajectory.tool_calls:
        parts.append(tc.name)
        if tc.arguments:
            for v in tc.arguments.values():
                parts.append(str(v))
        if tc.result:
            parts.append(tc.result)
        if tc.error:
            parts.append(tc.error)
    return " ".join(parts)


def _any_token_in_corpus(tokens: list[str], corpus: str) -> bool:
    """Check if any of the tokens appear in the corpus.

    Uses word-boundary matching for short tokens (<8 chars) to avoid
    false positives like "test" matching "latest". Longer tokens
    (commands, paths) use substring matching since they're specific enough.
    """
    corpus_lower = corpus.lower()
    for token in tokens:
        if not token:
            continue
        t = token.lower()
        if len(t) < 8:
            # Word-boundary match for short tokens
            if re.search(r"\b" + re.escape(t) + r"\b", corpus_lower):
                return True
        else:
            # Substring match for longer tokens (commands, paths)
            if t in corpus_lower:
                return True
    return False


# ------------------------------------------------------------------
# Correctness: did following the steps solve the problem?
# ------------------------------------------------------------------


@dataclass
class CorrectnessScore:
    verdict: str  # "correct" | "incorrect" | "needs_review"
    confidence: str  # "auto" | "human"
    reason: str


def score_correctness(
    adherence: AdherenceScore,
    outcome: Outcome,
) -> CorrectnessScore:
    """Determine correctness from adherence and session outcome.

    Conservative: only auto-labels when signals are clear.
    Everything else goes to human review.
    """
    if adherence.score >= 0.5 and outcome == Outcome.SUCCESS:
        return CorrectnessScore(
            verdict="correct",
            confidence="auto",
            reason="steps followed and session succeeded",
        )
    if adherence.score >= 0.5 and outcome == Outcome.FAILURE:
        return CorrectnessScore(
            verdict="incorrect",
            confidence="auto",
            reason="steps followed but session failed",
        )
    if adherence.score < 0.5 and outcome == Outcome.SUCCESS:
        return CorrectnessScore(
            verdict="needs_review",
            confidence="auto",
            reason="session succeeded but playbook was largely ignored",
        )
    if adherence.score < 0.5 and outcome == Outcome.FAILURE:
        return CorrectnessScore(
            verdict="needs_review",
            confidence="auto",
            reason="playbook ignored and session failed",
        )
    # outcome == UNKNOWN
    return CorrectnessScore(
        verdict="needs_review",
        confidence="auto",
        reason="outcome unknown",
    )
