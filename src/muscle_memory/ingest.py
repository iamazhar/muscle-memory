"""Harness-agnostic transcript and episode ingestion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome, Trajectory
from muscle_memory.outcomes import infer_outcome
from muscle_memory.personal_loop import add_measurement_for_task, capture_task

_TRANSCRIPT_HARNESS_BY_FORMAT = {
    "claude-jsonl": "claude-code",
    "codex-jsonl": "codex",
}


def episode_from_transcript(
    path: Path,
    transcript_format: str,
    *,
    project_path: str | None = None,
    prompt_override: str | None = None,
) -> Episode:
    from muscle_memory.harness import get_harness

    try:
        harness_name = _TRANSCRIPT_HARNESS_BY_FORMAT[transcript_format]
    except KeyError as exc:
        raise ValueError(f"Unknown transcript format: {transcript_format}") from exc

    trajectory = get_harness(harness_name).parse_transcript(path)
    _validate_transcript_signal(transcript_format, trajectory)

    if transcript_format == "codex-jsonl" and not prompt_override:
        raise ValueError(
            "codex-jsonl ingest requires --prompt because Codex logs do not preserve the original user prompt"
        )

    if prompt_override:
        trajectory.user_prompt = prompt_override
    return _episode_from_trajectory(
        trajectory,
        session_id=path.stem,
        project_path=project_path,
    )


def episode_from_json(path: Path) -> Episode:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Episode JSON must be an object")

    if "trajectory" not in data:
        raise ValueError("Episode JSON must include a 'trajectory' object")

    trajectory = Trajectory.model_validate(data["trajectory"])
    outcome = data.get("outcome")
    reward = data.get("reward")
    activated_skills = data.get("activated_skills") or []

    if outcome is None or reward is None:
        signal = infer_outcome(
            trajectory,
            user_followup=trajectory.user_followup,
            any_skills_activated=bool(activated_skills),
        )
        resolved_outcome = signal.outcome
        resolved_reward = signal.reward
    else:
        resolved_outcome = Outcome(str(outcome))
        resolved_reward = float(reward)

    return Episode(
        session_id=str(data.get("session_id") or "") or None,
        user_prompt=str(data.get("user_prompt") or trajectory.user_prompt or "(unknown)"),
        trajectory=trajectory,
        outcome=resolved_outcome,
        reward=resolved_reward,
        project_path=_optional_str(data.get("project_path")),
        activated_skills=_string_list(activated_skills),
    )


def ingest_transcript_file(
    path: Path,
    transcript_format: str,
    *,
    config: Config,
    store: Store,
    extract: bool = True,
    prompt_override: str | None = None,
) -> tuple[Episode, int]:
    episode = episode_from_transcript(
        path,
        transcript_format,
        project_path=str(config.project_root) if config.project_root is not None else None,
        prompt_override=prompt_override,
    )
    store.add_episode(episode)
    _record_measurement_for_episode(
        episode,
        store=store,
        harness=_TRANSCRIPT_HARNESS_BY_FORMAT[transcript_format],
    )
    added = extract_skills_for_episode(episode, config=config, store=store) if extract else 0
    return episode, added


def ingest_episode_file(
    path: Path, *, config: Config, store: Store, extract: bool = True
) -> tuple[Episode, int]:
    episode = episode_from_json(path)
    if episode.project_path is None and config.project_root is not None:
        episode.project_path = str(config.project_root)
    store.add_episode(episode)
    _record_measurement_for_episode(episode, store=store, harness=config.harness)
    added = extract_skills_for_episode(episode, config=config, store=store) if extract else 0
    return episode, added


def extract_skills_for_episode(episode: Episode, *, config: Config, store: Store) -> int:
    from muscle_memory.dedup import add_skill_with_dedup
    from muscle_memory.embeddings import make_embedder
    from muscle_memory.extractor import Extractor
    from muscle_memory.llm import make_llm

    llm = make_llm(config)
    embedder = make_embedder(config)
    extractor = Extractor(llm, config)
    skills = extractor.extract(episode)

    added = 0
    for skill in skills:
        was_added, _ = add_skill_with_dedup(store, embedder, skill)
        if was_added:
            added += 1
    return added


def _record_measurement_for_episode(
    episode: Episode,
    *,
    store: Store,
    harness: str,
) -> None:
    task = None
    if episode.session_id:
        task = store.find_latest_task_by_session(episode.session_id)
    if task is None:
        task = capture_task(
            store,
            raw_prompt=episode.user_prompt,
            cleaned_prompt=episode.user_prompt,
            harness=harness,
            project_path=episode.project_path,
            session_id=episode.session_id,
        )
    add_measurement_for_task(
        store,
        task=task,
        outcome=episode.outcome,
        reason="transcript ingest",
        input_tokens=episode.trajectory.input_tokens,
        output_tokens=episode.trajectory.output_tokens,
        injected_skill_tokens=_injected_tokens_for_task(store, task.id),
        tool_call_count=episode.trajectory.num_tool_calls(),
        comparable=episode.trajectory.num_tool_calls() > 0,
    )


def _injected_tokens_for_task(store: Store, task_id: str) -> int:
    activation_tokens = sum(
        activation.injected_token_count
        for activation in store.list_activations_for_task(task_id)
    )
    existing = store.get_measurement_for_task(task_id)
    existing_tokens = existing.injected_skill_tokens if existing is not None else 0
    return max(activation_tokens, existing_tokens)


def _validate_transcript_signal(transcript_format: str, trajectory: Trajectory) -> None:
    if trajectory.user_prompt.strip() or trajectory.assistant_turns or trajectory.tool_calls:
        return
    raise ValueError(
        f"Unsupported or low-signal {transcript_format} transcript: no prompt, assistant turns, or tool calls detected"
    )


def _episode_from_trajectory(
    trajectory: Trajectory,
    *,
    session_id: str | None,
    project_path: str | None,
    activated_skills: list[str] | None = None,
) -> Episode:
    activated = activated_skills or []
    signal = infer_outcome(
        trajectory,
        user_followup=trajectory.user_followup,
        any_skills_activated=bool(activated),
    )
    return Episode(
        session_id=session_id,
        user_prompt=trajectory.user_prompt or "(unknown)",
        trajectory=trajectory,
        outcome=signal.outcome,
        reward=signal.reward,
        project_path=project_path,
        activated_skills=activated,
    )


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
