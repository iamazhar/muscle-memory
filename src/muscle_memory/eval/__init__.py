"""Evaluation system for muscle-memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvalLabel:
    """A human-annotated ground-truth label for an episode or retrieval judgment."""

    id: str = ""
    label_type: str = "outcome"  # 'outcome' | 'retrieval'
    episode_id: str = ""

    # outcome labels
    heuristic_outcome: str = ""
    human_outcome: str = ""
    confidence: str = "high"  # 'high' | 'low'
    notes: str = ""

    # retrieval labels
    query_text: str = ""
    skill_id: str = ""
    relevance: str = ""  # 'relevant' | 'partial' | 'irrelevant'
    rank_position: int = 0

    labeled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
