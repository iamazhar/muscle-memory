"""Trajectory → candidate Skills.

Given a completed Episode, ask the LLM to extract reusable procedural
skills. The prompt extracts liberally; pruning and PPO refinement
manage quality over time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import resources
from typing import Any

from muscle_memory.config import Config
from muscle_memory.llm import LLM
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall


class ExtractionError(RuntimeError):
    """Raised when the LLM call or response parsing fails.

    Callers (bootstrap, stop hook) decide whether to propagate or
    swallow. Inside the extractor we never hide failures behind an
    empty list — that silently drops user signal and was a real bug
    we hit during dogfooding.
    """


def _load_prompt_template() -> str:
    return (
        resources.files("muscle_memory.prompts").joinpath("extract.md").read_text(encoding="utf-8")
    )


# We only elide very long trajectories — the "interesting" events
# (errors, retries, recoveries) often live in the middle, so aggressive
# head+tail sampling drops the very signal we care about.
#
# Rule of thumb from Sonnet 4.6's 200k context: a trajectory of ~200
# tool calls ≈ 30k tokens, which is fine. We only kick in at >400 calls.
MAX_TOOL_CALLS_BEFORE_ELISION = 400
MAX_TOOL_CALLS_KEPT_HEAD = 80
MAX_TOOL_CALLS_KEPT_TAIL = 120


def _format_tool_call(tc: ToolCall, idx: int) -> str:
    parts = [f"call-{idx} tool={tc.name}"]
    if tc.arguments:
        args_repr = ", ".join(f"{k}={_short(v)}" for k, v in tc.arguments.items())
        parts.append(f"  args: {args_repr}")
    if tc.result:
        parts.append(f"  result: {_short(tc.result, 300)}")
    if tc.error:
        parts.append(f"  ERROR: {_short(tc.error, 300)}")
    return "\n".join(parts)


def _short(value: Any, limit: int = 200) -> str:
    s = str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + f"... ({len(s) - limit} more chars)"


def _is_noise_prompt(prompt: str) -> bool:
    """Detect Claude Code slash-command / system wrappers at the start
    of a transcript so we don't report them as the user's goal."""
    if not prompt:
        return True
    low = prompt.strip().lower()
    if low.startswith("<local-command-"):
        return True
    if low.startswith("<command-name>") or low.startswith("<command-message>"):
        return True
    return False


def format_trajectory_for_extractor(episode: Episode) -> str:
    """Render an Episode as a compact block wrapped in XML tags for the extractor.

    The XML wrapper is deliberate: without clear structural boundaries,
    long trajectories caused Sonnet to "continue the document" instead
    of analyze it (observed during dogfooding).
    """
    traj = episode.trajectory

    # derive a clean user-goal line; fall back to empty if it's slash-command noise
    goal = episode.user_prompt if not _is_noise_prompt(episode.user_prompt) else ""
    if not goal and traj.assistant_turns:
        # first assistant turn often restates the goal
        goal = _short(traj.assistant_turns[0], 300)

    head: list[str] = []
    head.append("<episode>")
    head.append(f"<goal>{_short(goal, 500) or '(not captured)'}</goal>")
    head.append(f"<outcome>{episode.outcome.value}</outcome>")
    head.append(f"<total_tool_calls>{len(traj.tool_calls)}</total_tool_calls>")

    head.append("<trajectory>")

    if not traj.tool_calls:
        head.append("(no tool calls)")
    else:
        calls = traj.tool_calls
        if len(calls) <= MAX_TOOL_CALLS_BEFORE_ELISION:
            for i, tc in enumerate(calls):
                head.append(_format_tool_call(tc, i))
        else:
            head.append(f"-- first {MAX_TOOL_CALLS_KEPT_HEAD} tool calls --")
            for i, tc in enumerate(calls[:MAX_TOOL_CALLS_KEPT_HEAD]):
                head.append(_format_tool_call(tc, i))
            skipped = len(calls) - MAX_TOOL_CALLS_KEPT_HEAD - MAX_TOOL_CALLS_KEPT_TAIL
            head.append(f"\n-- [{skipped} tool calls elided for length] --\n")
            head.append(f"-- last {MAX_TOOL_CALLS_KEPT_TAIL} tool calls --")
            start = len(calls) - MAX_TOOL_CALLS_KEPT_TAIL
            for i, tc in enumerate(calls[start:], start=start):
                head.append(_format_tool_call(tc, i))

    head.append("</trajectory>")

    if traj.assistant_turns:
        head.append("<assistant_reasoning_sample>")
        for turn in traj.assistant_turns[-3:]:
            head.append(f"- {_short(turn, 300)}")
        head.append("</assistant_reasoning_sample>")

    head.append("</episode>")
    head.append("")
    head.append(
        "Now produce your JSON array. Remember: empty `[]` is a valid and "
        "often correct answer. Respond with the array and nothing else."
    )

    return "\n".join(head)


class Extractor:
    """LLM-driven candidate skill extractor."""

    def __init__(self, llm: LLM, config: Config):
        self.llm = llm
        self.config = config
        self._prompt_template = _load_prompt_template()

    def extract(self, episode: Episode) -> list[Skill]:
        """Return zero or a few candidate Skills derived from `episode`.

        Returns an empty list if the LLM legitimately judges nothing
        reusable was learned, or if the episode is a known FAILURE,
        or if the trajectory has no activity. Raises `ExtractionError`
        if the LLM call fails or its response can't be parsed —
        callers decide how to handle those.
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
        except Exception as exc:
            raise ExtractionError(f"LLM call failed: {exc}") from exc

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


__all__ = ["Extractor", "ExtractionError", "format_trajectory_for_extractor"]
