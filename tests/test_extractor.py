"""Tests for the skill extractor with a fake LLM."""

from __future__ import annotations

from typing import Any

from muscle_memory.config import Config
from muscle_memory.extractor import Extractor, format_trajectory_for_extractor
from muscle_memory.models import Episode, Outcome, Trajectory


class FakeLLM:
    def __init__(self, payload: Any):
        self.payload = payload
        self.model = "fake"
        self.last_user: str | None = None
        self.last_system: str | None = None

    def complete_text(self, system: str, user: str, **_: Any) -> str:
        self.last_system = system
        self.last_user = user
        return ""

    def complete_json(self, system: str, user: str, **_: Any) -> Any:
        self.last_system = system
        self.last_user = user
        return self.payload


def test_extractor_returns_skills_from_valid_payload(
    successful_episode: Episode, sample_config: Config
) -> None:
    llm = FakeLLM(
        [
            {
                "activation": "when pytest fails with ModuleNotFoundError",
                "execution": "1. look for tools/test-runner.sh",
                "termination": "tests pass",
                "tool_hints": ["Bash"],
                "tags": ["testing"],
            }
        ]
    )
    ex = Extractor(llm, sample_config)
    skills = ex.extract(successful_episode)
    assert len(skills) == 1
    assert skills[0].activation.startswith("when pytest")
    assert skills[0].source_episode_ids == [successful_episode.id]


def test_extractor_returns_empty_on_failed_episode(
    successful_trajectory: Trajectory, sample_config: Config
) -> None:
    ep = Episode(
        user_prompt="x",
        trajectory=successful_trajectory,
        outcome=Outcome.FAILURE,
    )
    ex = Extractor(FakeLLM([{"activation": "a", "execution": "b", "termination": "c"}]), sample_config)
    assert ex.extract(ep) == []


def test_extractor_respects_max_skills(
    successful_episode: Episode, sample_config: Config
) -> None:
    sample_config.extractor_max_skills_per_episode = 2
    llm = FakeLLM(
        [
            {"activation": f"skill {i}", "execution": "e", "termination": "t"}
            for i in range(5)
        ]
    )
    ex = Extractor(llm, sample_config)
    skills = ex.extract(successful_episode)
    assert len(skills) == 2


def test_extractor_tolerates_bad_payload(
    successful_episode: Episode, sample_config: Config
) -> None:
    ex = Extractor(FakeLLM("not a list"), sample_config)
    assert ex.extract(successful_episode) == []


def test_extractor_skips_invalid_skill_entries(
    successful_episode: Episode, sample_config: Config
) -> None:
    llm = FakeLLM(
        [
            {"activation": "valid", "execution": "e", "termination": "t"},
            {"activation": "", "execution": "e", "termination": "t"},  # invalid
            "not a dict",  # invalid
        ]
    )
    ex = Extractor(llm, sample_config)
    skills = ex.extract(successful_episode)
    assert len(skills) == 1
    assert skills[0].activation == "valid"


def test_format_trajectory_includes_errors(successful_episode: Episode) -> None:
    out = format_trajectory_for_extractor(successful_episode)
    assert "ModuleNotFoundError" in out
    assert "test-runner.sh" in out
    assert "5 passed" in out
    assert "OUTCOME: success" in out
