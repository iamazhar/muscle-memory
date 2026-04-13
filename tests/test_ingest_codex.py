"""Tests for Codex transcript ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.ingest import episode_from_transcript
from muscle_memory.models import Scope

runner = CliRunner()


CODEX_USER_PROMPT = "Update the OrbitOps hero so it sounds more decisive, then run the local checks."
CODEX_AGENT_PROMPT = (
    "I’m updating the OrbitOps hero so it sounds more decisive, then I’ll run the local checks "
    "so we can confirm nothing regressed."
)


def _make_config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
        harness="generic",
    )


def _write_codex_transcript(path: Path) -> None:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": CODEX_AGENT_PROMPT,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'sed -n \"100,150p\" ui.py'",
                "aggregated_output": "hero section source",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_2",
                "type": "agent_message",
                "text": "The smoke test hit a sandbox limit: this environment won't let check.py bind a local port.",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_3",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'python3 check.py'",
                "aggregated_output": "Traceback... PermissionError: [Errno 1] Operation not permitted",
                "exit_code": 1,
                "status": "failed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_4",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 python3 - <<PY ... PY'",
                "aggregated_output": "render validation passed",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")


def test_episode_from_codex_transcript_extracts_meaningful_signal_with_prompt_override(tmp_path: Path) -> None:
    transcript = tmp_path / "codex-task.jsonl"
    _write_codex_transcript(transcript)

    episode = episode_from_transcript(transcript, "codex-jsonl", prompt_override=CODEX_USER_PROMPT)

    assert episode.user_prompt == CODEX_USER_PROMPT
    assert len(episode.trajectory.assistant_turns) == 2
    assert episode.trajectory.num_tool_calls() == 3
    assert episode.trajectory.tool_calls[0].name == "Bash"
    assert episode.trajectory.tool_calls[1].error is not None
    assert "PermissionError" in episode.trajectory.tool_calls[1].error


def test_episode_from_codex_transcript_requires_prompt_override(tmp_path: Path) -> None:
    transcript = tmp_path / "codex-task.jsonl"
    _write_codex_transcript(transcript)

    with pytest.raises(ValueError, match="requires --prompt"):
        episode_from_transcript(transcript, "codex-jsonl")


def test_episode_from_codex_transcript_rejects_low_signal_logs(tmp_path: Path) -> None:
    transcript = tmp_path / "empty-codex.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "turn.started"}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="low-signal"):
        episode_from_transcript(transcript, "codex-jsonl")


def test_cli_ingest_codex_transcript_records_non_junk_episode(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    Store(cfg.db_path, embedding_dims=4)

    transcript = project_root / "codex-task.jsonl"
    _write_codex_transcript(transcript)

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(
            app,
            [
                "ingest",
                "transcript",
                str(transcript),
                "--format",
                "codex-jsonl",
                "--prompt",
                CODEX_USER_PROMPT,
                "--no-extract",
            ],
        )

    assert result.exit_code == 0
    store = Store(cfg.db_path, embedding_dims=4)
    episodes = store.list_episodes(limit=10)
    assert len(episodes) == 1
    assert episodes[0].user_prompt == CODEX_USER_PROMPT
    assert episodes[0].trajectory.num_tool_calls() == 3
    assert len(episodes[0].trajectory.assistant_turns) == 2
