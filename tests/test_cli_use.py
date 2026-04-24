"""Tests for the mm use command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.harness.claude_code import ClaudeCodeHarness
from muscle_memory.models import DeliveryMode, Maturity, Scope, Skill, TaskRecord
from muscle_memory.personal_loop import (
    count_text_tokens,
    format_context,
    record_activations,
)
from muscle_memory.retriever import RetrievedSkill

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
            1.0 if "docker" in tokens else 0.0,
            1.0 if "release" in tokens else 0.0,
        ]


def _make_config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
        harness="codex",
    )


def _seed_skill(store: Store, embedder: DummyEmbedder) -> Skill:
    skill = Skill(
        activation="When pytest fails with import errors",
        execution="1. Inspect the import path\n2. Run the targeted pytest command",
        termination="The failing pytest import error is resolved",
        maturity=Maturity.LIVE,
        successes=3,
        invocations=3,
        score=1.0,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill, embedding=embedder.embed_one(skill.activation))
    return skill


def _seed_skill_with_activation(
    store: Store,
    embedder: DummyEmbedder,
    activation: str,
    *,
    execution: str = "1. Do the practiced step",
    termination: str = "The task is complete",
) -> Skill:
    skill = Skill(
        activation=activation,
        execution=execution,
        termination=termination,
        maturity=Maturity.LIVE,
        successes=3,
        invocations=3,
        score=1.0,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill, embedding=embedder.embed_one(skill.activation))
    return skill


def test_use_outputs_context_and_records_activation(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["use", "pytest import errors"])

    assert result.exit_code == 0
    assert "<muscle_memory>" in result.output
    assert "pytest fails with import errors" in result.output

    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    assert tasks[0].cleaned_prompt == "pytest import errors"
    activations = store.list_activations_for_task(tasks[0].id)
    assert len(activations) == 1
    assert activations[0].skill_id == skill.id
    assert activations[0].delivery_mode is DeliveryMode.CODEX_USE
    assert activations[0].injected_token_count > 0


def test_use_json_reports_task_and_hits(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    _seed_skill(store, embedder)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["use", "pytest import errors", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["task"]["cleaned_prompt"] == "pytest import errors"
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["activation_id"]
    assert payload["hits"][0]["activation"]["id"] == payload["hits"][0]["activation_id"]
    assert payload["hits"][0]["activation"]["delivery_mode"] == DeliveryMode.CODEX_USE.value
    assert payload["hits"][0]["activation"]["injected_token_count"] > 0
    assert payload["context_token_count"] > 0


def test_use_json_reports_empty_hits_for_no_match(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["use", "unrelated task", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["context"] == ""
    assert payload["context_token_count"] == 0
    assert payload["hits"] == []


def test_use_records_task_even_when_no_skill_matches(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["use", "unrelated task"])

    assert result.exit_code == 0
    assert "No matching skills" in result.output
    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    assert store.list_activations_for_task(tasks[0].id) == []


def test_use_allocates_all_context_tokens_across_multiple_hits(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    _seed_skill_with_activation(store, embedder, "When pytest import errors block tests")
    _seed_skill_with_activation(store, embedder, "When pytest import failures happen")

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["use", "pytest import errors", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    tasks = store.list_tasks(limit=10)
    activations = store.list_activations_for_task(tasks[0].id)
    assert len(activations) == 2
    assert sum(a.injected_token_count for a in activations) == payload["context_token_count"]


def test_record_activations_allocates_remainder_tokens(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    store = Store(project_root / ".claude" / "mm.db", embedding_dims=4)
    embedder = DummyEmbedder()
    first = _seed_skill_with_activation(store, embedder, "When pytest import errors block tests")
    second = _seed_skill_with_activation(store, embedder, "When pytest import failures happen")
    task = TaskRecord(raw_prompt="pytest import errors", cleaned_prompt="pytest import errors")
    store.add_task(task)
    hits = [
        RetrievedSkill(skill=first, distance=0.1, score_bonus=0.0),
        RetrievedSkill(skill=second, distance=0.2, score_bonus=0.0),
    ]
    context_token_count = 5

    activations = record_activations(
        store,
        task=task,
        hits=hits,
        delivery_mode=DeliveryMode.CODEX_USE,
        context_token_count=context_token_count,
    )

    assert len(activations) == 2
    assert sum(a.injected_token_count for a in activations) == context_token_count


def test_use_does_not_mutate_skill_counters(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _make_config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)
    before = store.get_skill(skill.id)
    assert before is not None

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli.make_embedder", return_value=embedder),
    ):
        result = runner.invoke(app, ["use", "pytest import errors"])

    assert result.exit_code == 0
    after = store.get_skill(skill.id)
    assert after is not None
    assert after.invocations == before.invocations
    assert after.successes == before.successes
    assert after.failures == before.failures
    assert after.score == before.score
    assert after.maturity is before.maturity
    assert after.last_used_at == before.last_used_at


def test_claude_harness_uses_shared_context_formatter() -> None:
    skill = Skill(
        activation="When pytest fails with import errors",
        execution="1. Inspect the import path\n2. Run the targeted pytest command",
        termination="The failing pytest import error is resolved",
        maturity=Maturity.LIVE,
        successes=3,
        invocations=3,
        score=1.0,
    )
    hits = [RetrievedSkill(skill=skill, distance=0.1, score_bonus=0.0)]

    context = format_context(hits)

    assert ClaudeCodeHarness().format_context(hits) == context
    assert count_text_tokens(context) > 0
