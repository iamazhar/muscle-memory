"""Trajectory → candidate Skills.

Given a completed Episode, ask the LLM to extract zero or a few
reusable procedural skills. The prompt is intentionally conservative:
most episodes should yield zero skills.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import resources
from typing import Any

from muscle_memory.config import Config
from muscle_memory.llm import LLM
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall, Trajectory


def _load_prompt_template() -> str:
    return (
        resources.files("muscle_memory.prompts")
        .joinpath("extract.md")
        .read_text(encoding="utf-8")
    )


def _format_tool_call(tc: ToolCall, idx: int) -> str:
    parts = [f"[{idx}] tool={tc.name}"]
    if tc.arguments:
        # truncate long argument values
        args_repr = ", ".join(f"{k}={_short(v)}" for k, v in tc.arguments.items())
        parts.append(f"  args: {args_repr}")
    if tc.result:
        parts.append(f"  result: {_short(tc.result, 400)}")
    if tc.error:
        parts.append(f"  ERROR: {_short(tc.error, 400)}")
    return "\n".join(parts)


def _short(value: Any, limit: int = 200) -> str:
    s = str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + f"... ({len(s) - limit} more chars)"


def format_trajectory_for_extractor(episode: Episode) -> str:
    """Render an Episode as a compact text block for the extractor."""
    lines: list[str] = []
    lines.append(f"USER PROMPT: {episode.user_prompt}")
    lines.append(f"OUTCOME: {episode.outcome.value}")
    if episode.activated_skills:
        lines.append(
            f"SKILLS THAT WERE ACTIVE: {', '.join(episode.activated_skills)}"
        )
    lines.append("")
    lines.append("TRAJECTORY:")

    traj = episode.trajectory
    if not traj.tool_calls and not traj.assistant_turns:
        lines.append("(no tool calls)")
    else:
        # interleave assistant turns and tool calls in original order
        for i, tc in enumerate(traj.tool_calls):
            lines.append(_format_tool_call(tc, i))
        if traj.assistant_turns:
            lines.append("")
            lines.append("ASSISTANT REASONING:")
            for turn in traj.assistant_turns[-5:]:  # last 5 only
                lines.append(f"- {_short(turn, 300)}")

    return "\n".join(lines)


class Extractor:
    """LLM-driven candidate skill extractor."""

    def __init__(self, llm: LLM, config: Config):
        self.llm = llm
        self.config = config
        self._prompt_template = _load_prompt_template()

    def extract(self, episode: Episode) -> list[Skill]:
        """Return zero or a few candidate Skills derived from `episode`.

        Returns an empty list if the LLM judges nothing reusable was
        learned, if the episode failed, or if the LLM response can't
        be parsed into valid Skill objects.
        """
        if episode.outcome is Outcome.FAILURE:
            return []

        if not episode.trajectory.tool_calls and not episode.trajectory.assistant_turns:
            return []

        system = self._prompt_template.replace(
            "{max_skills}", str(self.config.extractor_max_skills_per_episode)
        )
        user = format_trajectory_for_extractor(episode)

        try:
            raw = self.llm.complete_json(system, user, max_tokens=2048, temperature=0.2)
        except Exception:
            # extraction is best-effort; never blow up the caller
            return []

        return self._coerce_skills(raw, source_episode_id=episode.id)

    def _coerce_skills(self, raw: Any, *, source_episode_id: str) -> list[Skill]:
        if not isinstance(raw, list):
            return []
        out: list[Skill] = []
        now = datetime.now(UTC)
        for item in raw[: self.config.extractor_max_skills_per_episode]:
            if not isinstance(item, dict):
                continue
            try:
                skill = Skill(
                    activation=str(item.get("activation", "")),
                    execution=str(item.get("execution", "")),
                    termination=str(item.get("termination", "")),
                    tool_hints=_as_str_list(item.get("tool_hints")),
                    tags=_as_str_list(item.get("tags")),
                    scope=Scope.PROJECT,
                    source_episode_ids=[source_episode_id],
                    created_at=now,
                )
            except Exception:
                continue
            out.append(skill)
        return out


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return []


__all__ = ["Extractor", "format_trajectory_for_extractor"]
