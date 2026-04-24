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


def test_config_load_reads_harness_from_project_config(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "mm.json").write_text(json.dumps({"harness": "codex"}), encoding="utf-8")

    cfg = Config.load(start_dir=tmp_path)

    assert cfg.harness == "codex"


def test_config_env_harness_overrides_project_config(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "mm.json").write_text(
        json.dumps({"harness": "claude-code"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MM_HARNESS", "codex")

    cfg = Config.load(start_dir=tmp_path)

    assert cfg.harness == "codex"


def test_init_requires_explicit_harness_when_not_interactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)

    with patch("muscle_memory.cli._stdin_is_interactive", return_value=False, create=True):
        result = runner.invoke(app, ["init"])

    assert result.exit_code == 1
    assert "Pass --harness" in result.output


def test_init_prompts_for_harness_when_interactive(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)

    with patch("muscle_memory.cli._stdin_is_interactive", return_value=True, create=True):
        result = runner.invoke(app, ["init"], input="codex\n")

    assert result.exit_code == 0
    assert "Harness: codex" in result.output
    config_path = tmp_path / ".claude" / "mm.json"
    assert config_path.exists()
    assert json.loads(config_path.read_text(encoding="utf-8"))["harness"] == "codex"


def test_init_outside_project_does_not_prompt_for_harness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with (
        patch("muscle_memory.cli._stdin_is_interactive", return_value=True, create=True),
        patch("muscle_memory.cli.Prompt.ask", return_value="codex") as ask,
    ):
        result = runner.invoke(app, ["init"])

    assert result.exit_code == 1
    assert "Not inside a project" in result.output
    ask.assert_not_called()


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


def test_learn_transcript_records_episode_without_extraction(tmp_path: Path) -> None:
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
            [
                "learn",
                "--transcript",
                str(transcript),
                "--format",
                "claude-jsonl",
                "--no-extract",
            ],
        )

    assert result.exit_code == 0
    store = Store(cfg.db_path, embedding_dims=4)
    episodes = store.list_episodes(limit=10)
    assert len(episodes) == 1
    assert episodes[0].user_prompt == "run the tests"
