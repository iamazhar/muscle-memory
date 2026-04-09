"""Core data models for muscle-memory.

A Skill is the central unit of procedural memory. It is, per ProcMEM,
a natural-language tuple ⟨activation, execution, termination⟩ that the
agent reads and executes with judgment — no DSL, no code templates.

Episodes record completed agent trajectories so that scores can be
updated and new Skills extracted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex[:12]


class Maturity(str, Enum):
    """How trusted a Skill is, based on how often it's been useful."""

    CANDIDATE = "candidate"  # fresh, needs validation
    ESTABLISHED = "established"  # proven a few times
    PROVEN = "proven"  # reliably useful


class Outcome(str, Enum):
    """Inferred or reported outcome of an episode."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Scope(str, Enum):
    """Where a skill is visible — per-project or user-wide."""

    PROJECT = "project"
    GLOBAL = "global"


class ToolCall(BaseModel):
    """A single tool invocation within a trajectory."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: int | None = None

    def is_error(self) -> bool:
        return self.error is not None


class Trajectory(BaseModel):
    """A sequence of turns and tool calls during a task.

    Represented flatly as an ordered list of events so that the
    extractor can reason about call order and causality.
    """

    user_prompt: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    assistant_turns: list[str] = Field(default_factory=list)

    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    def errored_tool_calls(self) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.is_error()]

    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]


class Skill(BaseModel):
    """A reusable procedural playbook.

    All three text fields are human-readable and user-editable.
    Embeddings are derived from `activation` and cached in the DB
    but are never canonical — the text is the source of truth.
    """

    id: str = Field(default_factory=_new_id)

    # the core ProcMEM tuple
    activation: str = Field(description="Natural-language description of when this Skill applies.")
    execution: str = Field(
        description="Ordered steps the agent should take while this Skill is active."
    )
    termination: str = Field(
        description="Natural-language condition under which the Skill has finished."
    )

    # light-touch structure on top of the text fields
    tool_hints: list[str] = Field(
        default_factory=list,
        description="Optional hints about which tools to prefer, e.g. 'Grep not Bash'.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags: language, framework, task type.",
    )
    scope: Scope = Scope.PROJECT

    # scoring and provenance
    score: float = 0.0
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    maturity: Maturity = Maturity.CANDIDATE
    source_episode_ids: list[str] = Field(default_factory=list)

    # v0.2 — refinement state
    refinement_count: int = 0
    previous_text: dict[str, str] | None = Field(
        default=None,
        description="Most recent pre-refinement state, for rollback. "
        "Keys: activation, execution, termination.",
    )

    # timestamps
    created_at: datetime = Field(default_factory=_now)
    last_used_at: datetime | None = None
    last_refined_at: datetime | None = None

    @field_validator("activation", "execution", "termination")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    def recompute_score(self) -> None:
        """Score = successes / invocations, clamped to [0, 1].

        An unused Skill has score 0. A Skill with only failures has score 0.
        """
        if self.invocations == 0:
            self.score = 0.0
        else:
            self.score = self.successes / self.invocations

    def recompute_maturity(self) -> None:
        """Promote Skills that have proven themselves.

        Rules of thumb:
          * < 3 successes → candidate
          * 3–9 successes, score ≥ 0.6 → established
          * ≥ 10 successes, score ≥ 0.7 → proven
        """
        if self.successes >= 10 and self.score >= 0.7:
            self.maturity = Maturity.PROVEN
        elif self.successes >= 3 and self.score >= 0.6:
            self.maturity = Maturity.ESTABLISHED
        else:
            self.maturity = Maturity.CANDIDATE


class Episode(BaseModel):
    """A completed (or in-progress) agent task, used for scoring and extraction."""

    id: str = Field(default_factory=_new_id)
    session_id: str | None = None

    user_prompt: str
    trajectory: Trajectory
    outcome: Outcome = Outcome.UNKNOWN
    reward: float = 0.0

    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None

    project_path: str | None = None

    # which skills were active during this episode (by id)
    activated_skills: list[str] = Field(default_factory=list)
