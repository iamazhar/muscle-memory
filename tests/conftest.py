"""Shared pytest fixtures for muscle-memory tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome, Scope, Skill, ToolCall, Trajectory


@pytest.fixture
def tmp_db(tmp_path: Path) -> Store:
    """A small SQLite store at a temp path with 4-dim embeddings (fast)."""
    db_path = tmp_path / "mm.db"
    return Store(db_path, embedding_dims=4)


@pytest.fixture
def sample_skill() -> Skill:
    return Skill(
        activation="When pytest fails with ModuleNotFoundError in this monorepo",
        execution="1. Check tools/test-runner.sh\n2. Use it instead of pytest",
        termination="Tests pass or runner is missing",
        tool_hints=["Bash: tools/test-runner.sh"],
        tags=["testing", "python"],
        scope=Scope.PROJECT,
    )


@pytest.fixture
def successful_trajectory() -> Trajectory:
    return Trajectory(
        user_prompt="run the tests",
        tool_calls=[
            ToolCall(
                name="Bash",
                arguments={"command": "pytest"},
                error="ModuleNotFoundError: No module named 'myapp'",
            ),
            ToolCall(
                name="Bash",
                arguments={"command": "ls tools/"},
                result="test-runner.sh\nbuild.sh",
            ),
            ToolCall(
                name="Bash",
                arguments={"command": "./tools/test-runner.sh"},
                result="5 passed in 2.13s",
            ),
        ],
        assistant_turns=["trying pytest", "found test-runner.sh", "using it"],
    )


@pytest.fixture
def successful_episode(successful_trajectory: Trajectory) -> Episode:
    return Episode(
        user_prompt="run the tests",
        trajectory=successful_trajectory,
        outcome=Outcome.SUCCESS,
        reward=0.6,
    )


@pytest.fixture
def sample_config(tmp_path: Path) -> Config:
    return Config(
        db_path=tmp_path / "mm.db",
        scope=Scope.PROJECT,
        project_root=tmp_path,
        embedding_dims=4,
        extractor_max_skills_per_episode=3,
    )
