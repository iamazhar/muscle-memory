"""Tests for transcript token measurement."""

from __future__ import annotations

import json
from pathlib import Path

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.harness.claude_code import ClaudeCodeHarness
from muscle_memory.harness.codex import CodexHarness
from muscle_memory.ingest import ingest_episode_file, ingest_transcript_file
from muscle_memory.models import (
    ActivationRecord,
    DeliveryMode,
    MeasurementRecord,
    Outcome,
    Scope,
)
from muscle_memory.personal_loop import capture_task


def _config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
        harness="codex",
    )


def test_codex_parser_captures_turn_usage(tmp_path: Path) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "I will run checks."},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "pytest",
                            "aggregated_output": "5 passed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 1200, "output_tokens": 300},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 25, "output_tokens": 5},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    trajectory = CodexHarness().parse_transcript(transcript)

    assert trajectory.input_tokens == 1225
    assert trajectory.output_tokens == 305


def test_claude_parser_captures_turn_usage(tmp_path: Path) -> None:
    transcript = tmp_path / "claude.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "run pytest"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "usage": {"input_tokens": 100, "output_tokens": 20},
                            "content": [{"type": "text", "text": "Running tests."}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "usage": {"input_tokens": 50, "output_tokens": 10},
                        "message": {
                            "content": [{"type": "text", "text": "Tests passed."}]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    trajectory = ClaudeCodeHarness().parse_transcript(transcript)

    assert trajectory.input_tokens == 150
    assert trajectory.output_tokens == 30


def test_ingest_transcript_creates_task_and_measurement(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    transcript = project_root / "codex.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "I will run checks."},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "pytest",
                            "aggregated_output": "5 passed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 1200, "output_tokens": 300},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    episode, added = ingest_transcript_file(
        transcript,
        "codex-jsonl",
        config=cfg,
        store=store,
        extract=False,
        prompt_override="run pytest",
    )

    assert added == 0
    assert episode.trajectory.input_tokens == 1200
    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    assert tasks[0].harness == "codex"
    measurement = store.get_measurement_for_task(tasks[0].id)
    assert measurement is not None
    assert measurement.input_tokens == 1200
    assert measurement.output_tokens == 300
    assert measurement.comparable is True


def test_ingest_episode_creates_task_and_measurement(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    episode = project_root / "episode.json"
    episode.write_text(
        json.dumps(
            {
                "session_id": "episode-session",
                "user_prompt": "run pytest",
                "trajectory": {
                    "user_prompt": "run pytest",
                    "assistant_turns": ["Running pytest."],
                    "tool_calls": [
                        {
                            "name": "Bash",
                            "arguments": {"command": "pytest"},
                            "result": "5 passed",
                        }
                    ],
                    "input_tokens": 400,
                    "output_tokens": 90,
                },
                "outcome": "success",
                "reward": 1.0,
            }
        ),
        encoding="utf-8",
    )

    ingest_episode_file(episode, config=cfg, store=store, extract=False)

    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    assert tasks[0].harness == "codex"
    measurement = store.get_measurement_for_task(tasks[0].id)
    assert measurement is not None
    assert measurement.input_tokens == 400
    assert measurement.output_tokens == 90
    assert measurement.comparable is True


def test_ingest_preserves_existing_injected_token_evidence(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    task = capture_task(
        store,
        raw_prompt="run pytest",
        cleaned_prompt="run pytest",
        harness="codex",
        project_path=str(project_root),
        session_id="session-1",
    )
    store.add_activation(
        ActivationRecord(
            task_id=task.id,
            skill_id="skill-1",
            delivery_mode=DeliveryMode.CODEX_USE,
            injected_token_count=77,
        )
    )
    store.add_measurement(
        MeasurementRecord(
            task_id=task.id,
            outcome=Outcome.UNKNOWN,
            injected_skill_tokens=77,
        )
    )
    transcript = project_root / "session-1.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Running tests."},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "pytest",
                            "aggregated_output": "5 passed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 200, "output_tokens": 20},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    ingest_transcript_file(
        transcript,
        "codex-jsonl",
        config=cfg,
        store=store,
        extract=False,
        prompt_override="run pytest",
    )

    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    measurement = store.get_measurement_for_task(task.id)
    assert measurement is not None
    assert measurement.input_tokens == 200
    assert measurement.output_tokens == 20
    assert measurement.injected_skill_tokens == 77
