"""Tests for candidate review commands."""

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


def _make_config(store_dir: Path) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
    )


def test_review_list_shows_candidates_only(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    candidate = Skill(
        activation="When pytest import fails",
        execution="1. inspect import path\n2. run test runner",
        termination="tests pass",
        maturity=Maturity.CANDIDATE,
        source_episode_ids=["ep1", "ep2"],
    )
    live = Skill(
        activation="When git push is rejected",
        execution="1. pull --rebase\n2. push again",
        termination="push succeeds",
        maturity=Maturity.LIVE,
    )
    store.add_skill(candidate)
    store.add_skill(live)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["review", "list", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["id"] == candidate.id
    assert data[0]["source_evidence"] == 2


def test_review_approve_promotes_candidate(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    skill = Skill(
        activation="When pytest import fails",
        execution="1. inspect import path\n2. run test runner",
        termination="tests pass",
        maturity=Maturity.CANDIDATE,
    )
    store.add_skill(skill)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["review", "approve", skill.id])

    assert result.exit_code == 0
    updated = store.get_skill(skill.id)
    assert updated is not None
    assert updated.maturity is Maturity.LIVE


def test_review_reject_deletes_skill(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)

    skill = Skill(
        activation="When pytest import fails",
        execution="1. inspect import path\n2. run test runner",
        termination="tests pass",
        maturity=Maturity.CANDIDATE,
    )
    store.add_skill(skill)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["review", "reject", skill.id])

    assert result.exit_code == 0
    assert store.get_skill(skill.id) is None

