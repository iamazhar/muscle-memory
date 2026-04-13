"""Harness-agnostic transcript and episode ingestion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome, Trajectory
from muscle_memory.outcomes import infer_outcome


def episode_from_transcript(path: Path, transcript_format: str, *, project_path: str | None = None) -> Episode:
    if transcript_format == "claude-jsonl":
        from muscle_memory.harness import get_harness

        trajectory = get_harness("claude-code").parse_transcript(path)
        return _episode_from_trajectory(
            trajectory,
            session_id=path.stem,
            project_path=project_path,
        )
    raise ValueError(f"Unknown transcript format: {transcript_format}")


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
) -> tuple[Episode, int]:
    episode = episode_from_transcript(
        path,
        transcript_format,
        project_path=str(config.project_root) if config.project_root is not None else None,
    )
    store.add_episode(episode)
    added = extract_skills_for_episode(episode, config=config, store=store) if extract else 0
    return episode, added


def ingest_episode_file(path: Path, *, config: Config, store: Store, extract: bool = True) -> tuple[Episode, int]:
    episode = episode_from_json(path)
    if episode.project_path is None and config.project_root is not None:
        episode.project_path = str(config.project_root)
    store.add_episode(episode)
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
