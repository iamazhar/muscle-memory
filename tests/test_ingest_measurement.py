"""Tests for transcript token measurement."""

from __future__ import annotations

import json
from pathlib import Path

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.harness.codex import CodexHarness
from muscle_memory.ingest import ingest_transcript_file
from muscle_memory.models import Scope


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
            ]
        ),
        encoding="utf-8",
    )

    trajectory = CodexHarness().parse_transcript(transcript)

    assert trajectory.input_tokens == 1200
    assert trajectory.output_tokens == 300


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
    measurement = store.get_measurement_for_task(tasks[0].id)
    assert measurement is not None
    assert measurement.input_tokens == 1200
    assert measurement.output_tokens == 300
    assert measurement.comparable is True
