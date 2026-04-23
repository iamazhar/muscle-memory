"""Tests for harness-aware init, retrieval, and ingest commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Maturity, Scope, Skill

runner = CliRunner()


class DummyEmbedder:
    dims = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        tokens = text.lower()
        return [
            1.0 if "pytest" in tokens else 0.0,
            1.0 if "import" in tokens else 0.0,
            1.0 if "rails" in tokens else 0.0,
            1.0 if "docker" in tokens else 0.0,
        ]


def _make_config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
        harness="generic",
    )


def test_config_load_reads_harness_from_env(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("MM_HARNESS", "generic")

    cfg = Config.load(start_dir=tmp_path)

    assert cfg.harness == "generic"


def test_retrieve_returns_matching_skills_as_json(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()

    skill = Skill(
        activation="When pytest fails with import errors",
        execution="1. inspect import path\n2. rerun targeted tests",
        termination="tests pass",
        maturity=Maturity.LIVE,
        successes=2,
        invocations=2,
        score=1.0,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill, embedding=embedder.embed_one(skill.activation))

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["retrieve", "pytest import errors", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload) == 1
    assert payload[0]["id"] == skill.id
    assert payload[0]["activation"] == skill.activation
    assert "distance" in payload[0]


def test_ingest_transcript_records_episode_without_extraction(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    Store(cfg.db_path, embedding_dims=4)

    transcript = project_root / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "run the tests"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Trying pytest"},
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": "pytest"},
                                },
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t1",
                                    "content": "5 passed",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(
            app,
            ["ingest", "transcript", str(transcript), "--format", "claude-jsonl", "--no-extract"],
        )

    assert result.exit_code == 0
    store = Store(cfg.db_path, embedding_dims=4)
    episodes = store.list_episodes(limit=10)
    assert len(episodes) == 1
    assert episodes[0].user_prompt == "run the tests"
