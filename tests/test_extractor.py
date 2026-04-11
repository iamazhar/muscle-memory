"""Tests for the skill extractor with a fake LLM."""

from __future__ import annotations

from typing import Any

import pytest

from muscle_memory.config import Config
from muscle_memory.extractor import (
    ExtractionError,
    Extractor,
    format_trajectory_for_extractor,
)
from muscle_memory.models import Episode, Outcome, ToolCall, Trajectory


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
                "execution": "1. look for tools/test-runner.sh\n2. run it instead of pytest",
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
    ex = Extractor(
        FakeLLM([{"activation": "a", "execution": "b", "termination": "c"}]), sample_config
    )
    assert ex.extract(ep) == []


def test_extractor_returns_empty_on_unknown_episode(
    successful_trajectory: Trajectory, sample_config: Config
) -> None:
    ep = Episode(
        user_prompt="x",
        trajectory=successful_trajectory,
        outcome=Outcome.UNKNOWN,
    )
    ex = Extractor(
        FakeLLM(
            [
                {
                    "activation": "when pytest fails with ModuleNotFoundError",
                    "execution": "1. look for tools/test-runner.sh\n2. run it",
                    "termination": "tests pass",
                }
            ]
        ),
        sample_config,
    )
    assert ex.extract(ep) == []


def test_extractor_returns_empty_on_tiny_success_episode(sample_config: Config) -> None:
    ep = Episode(
        user_prompt="x",
        trajectory=Trajectory(
            tool_calls=[ToolCall(name="Bash", arguments={"command": "pwd"}, result="/tmp/project")]
        ),
        outcome=Outcome.SUCCESS,
    )
    ex = Extractor(
        FakeLLM(
            [
                {
                    "activation": "when checking the working directory",
                    "execution": "1. run pwd\n2. continue",
                    "termination": "cwd is known",
                }
            ]
        ),
        sample_config,
    )
    assert ex.extract(ep) == []


def test_extractor_respects_max_skills(successful_episode: Episode, sample_config: Config) -> None:
    sample_config.extractor_max_skills_per_episode = 2
    llm = FakeLLM(
        [
            {
                "activation": f"When reusable failure pattern {i} appears",
                "execution": "1. inspect the failing command\n2. apply the known fix",
                "termination": "the command succeeds",
            }
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


def test_extractor_raises_on_llm_failure(
    successful_episode: Episode, sample_config: Config
) -> None:
    """LLM failures must be loud, not silent. Regression test for
    the bug where bootstrap reported 'skills: 0' while an API-credit
    error was silently swallowed."""

    class ExplodingLLM:
        model = "boom"

        def complete_text(self, *a, **k):
            raise RuntimeError("boom")

        def complete_json(self, *a, **k):
            raise RuntimeError("boom")

    ex = Extractor(ExplodingLLM(), sample_config)
    with pytest.raises(ExtractionError, match="LLM call failed"):
        ex.extract(successful_episode)


def test_extractor_skips_invalid_skill_entries(
    successful_episode: Episode, sample_config: Config
) -> None:
    llm = FakeLLM(
        [
            {
                "activation": "valid reusable trigger",
                "execution": "1. do x\n2. do y",
                "termination": "t",
            },
            {"activation": "", "execution": "1. do x\n2. do y", "termination": "t"},  # invalid
            "not a dict",  # invalid
        ]
    )
    ex = Extractor(llm, sample_config)
    skills = ex.extract(successful_episode)
    assert len(skills) == 1
    assert skills[0].activation == "valid reusable trigger"


def test_extractor_rejects_one_off_literals(
    successful_episode: Episode, sample_config: Config
) -> None:
    llm = FakeLLM(
        [
            {
                "activation": "When PR #482 fails on 2026-04-11 for session 123e4567-e89b-12d3-a456-426614174000",
                "execution": "1. Inspect /private/tmp/build.log\n2. Retry the job",
                "termination": "The exact PR run passes",
            }
        ]
    )
    ex = Extractor(llm, sample_config)
    assert ex.extract(successful_episode) == []


def test_extractor_rejects_single_step_skills(
    successful_episode: Episode, sample_config: Config
) -> None:
    llm = FakeLLM(
        [
            {
                "activation": "When pytest fails with ModuleNotFoundError",
                "execution": "1. run the test runner",
                "termination": "tests pass",
            }
        ]
    )
    ex = Extractor(llm, sample_config)
    assert ex.extract(successful_episode) == []


def test_format_trajectory_includes_errors(successful_episode: Episode) -> None:
    out = format_trajectory_for_extractor(successful_episode)
    assert "ModuleNotFoundError" in out
    assert "test-runner.sh" in out
    assert "5 passed" in out
    assert "<outcome>success</outcome>" in out
    assert "<trajectory>" in out
    assert "</trajectory>" in out


def test_format_trajectory_elides_only_very_long_histories() -> None:
    """Elision kicks in above MAX_TOOL_CALLS_BEFORE_ELISION; shorter trajectories
    stay intact because the juicy recovery/error events often live in the middle."""
    from muscle_memory.extractor import (
        MAX_TOOL_CALLS_BEFORE_ELISION,
        MAX_TOOL_CALLS_KEPT_HEAD,
    )
    from muscle_memory.models import Episode, Outcome, ToolCall, Trajectory

    # just over the threshold
    n = MAX_TOOL_CALLS_BEFORE_ELISION + 50
    calls = [
        ToolCall(name="Bash", arguments={"command": f"echo {i}"}, result=str(i)) for i in range(n)
    ]
    ep = Episode(
        user_prompt="x",
        trajectory=Trajectory(tool_calls=calls),
        outcome=Outcome.SUCCESS,
    )
    out = format_trajectory_for_extractor(ep)
    assert "elided for length" in out
    assert "call-0 " in out
    assert f"call-{n - 1} " in out
    mid = MAX_TOOL_CALLS_KEPT_HEAD + 20
    assert f"call-{mid} " not in out


def test_format_trajectory_keeps_medium_histories_whole() -> None:
    """Trajectories under the threshold should NOT be elided — the
    interesting events often live in the middle of a recovery."""
    from muscle_memory.extractor import MAX_TOOL_CALLS_BEFORE_ELISION
    from muscle_memory.models import Episode, Outcome, ToolCall, Trajectory

    n = MAX_TOOL_CALLS_BEFORE_ELISION - 1
    calls = [
        ToolCall(name="Bash", arguments={"command": f"echo {i}"}, result=str(i)) for i in range(n)
    ]
    ep = Episode(
        user_prompt="x",
        trajectory=Trajectory(tool_calls=calls),
        outcome=Outcome.SUCCESS,
    )
    out = format_trajectory_for_extractor(ep)
    assert "elided for length" not in out
    # every call should be present
    assert "call-0 " in out
    assert f"call-{n // 2} " in out
    assert f"call-{n - 1} " in out


def test_user_prompt_hook_skips_shell_escape_commands() -> None:
    """Bang prefixes, slash commands, and single-token shell commands
    should not trigger skill retrieval."""
    from muscle_memory.hooks.user_prompt import _is_shell_escape

    # bang prefix
    assert _is_shell_escape("!mm list")
    assert _is_shell_escape("!git status")
    # slash command
    assert _is_shell_escape("/model")
    assert _is_shell_escape("/clear")
    # bare shell command
    assert _is_shell_escape("mm list")
    assert _is_shell_escape("ls src/")
    assert _is_shell_escape("git log --oneline")
    assert _is_shell_escape("python3 --version")
    # natural-language prompts are NOT escaped
    assert not _is_shell_escape("help me fix this test")
    assert not _is_shell_escape("My python package isn't importing")
    assert not _is_shell_escape("Can you run pytest and show me the output?")
    # empty is technically escape (skip retrieval)
    assert _is_shell_escape("")
    assert _is_shell_escape("   ")


def test_format_trajectory_drops_slash_command_noise_from_goal() -> None:
    """The extractor should not treat /model caveat text as the user's goal."""
    from muscle_memory.models import Episode, ToolCall, Trajectory

    ep = Episode(
        user_prompt="<local-command-caveat>Caveat: slash command stuff</local-command-caveat>",
        trajectory=Trajectory(
            tool_calls=[ToolCall(name="Bash", arguments={"command": "ls"}, result="x")],
            assistant_turns=["The user wants me to list files."],
        ),
    )
    out = format_trajectory_for_extractor(ep)
    assert "local-command-caveat" not in out.split("<trajectory>")[0]
    # should fall back to the assistant's rephrasing
    assert "list files" in out
