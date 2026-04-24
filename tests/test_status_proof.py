"""Tests for proof-oriented status metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import (
    ActivationRecord,
    DeliveryMode,
    EvidenceConfidence,
    MeasurementRecord,
    Outcome,
    Scope,
    TaskRecord,
)
from muscle_memory.personal_loop import compute_proof_metrics

runner = CliRunner()


def _config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
    )


def _task(
    store: Store,
    prompt: str,
    *,
    assisted: bool,
    outcome: Outcome,
    tokens: int | None,
    output_tokens: int | None = 100,
    comparable: bool = True,
) -> None:
    task = TaskRecord(
        raw_prompt=prompt,
        cleaned_prompt=prompt,
        harness="codex",
        project_path="/repo",
    )
    store.add_task(task)
    if assisted:
        store.add_activation(
            ActivationRecord(
                task_id=task.id,
                skill_id="skill-1",
                delivery_mode=DeliveryMode.CODEX_USE,
                injected_token_count=25,
            )
        )
    store.add_measurement(
        MeasurementRecord(
            task_id=task.id,
            outcome=outcome,
            confidence=EvidenceConfidence.HIGH,
            reason="test fixture",
            input_tokens=tokens,
            output_tokens=output_tokens,
            injected_skill_tokens=25 if assisted else 0,
            tool_call_count=2,
            comparable=comparable,
        )
    )


def test_status_json_includes_proof_metrics(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    for index in range(5):
        _task(store, f"assisted {index}", assisted=True, outcome=Outcome.SUCCESS, tokens=700)
    for index in range(5):
        _task(store, f"unassisted {index}", assisted=False, outcome=Outcome.FAILURE, tokens=1000)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    proof = payload["proof"]
    assert proof["confidence"] == "medium"
    assert proof["comparable_tasks"] == 10
    assert proof["assisted_tasks"] == 5
    assert proof["unassisted_tasks"] == 5
    assert proof["assisted_success_rate"] == 1.0
    assert proof["unassisted_success_rate"] == 0.0
    assert proof["outcome_lift"] == 1.0
    assert proof["token_reduction"] == pytest.approx(3 / 11)
    assert proof["token_samples"] == 10
    assert proof["unknown_outcomes"] == 0


def test_status_rich_output_handles_insufficient_evidence(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Proof" in result.output
    assert "insufficient evidence" in result.output
    assert "Need at least 10 comparable measured tasks" in result.output
    assert "No skills yet" in result.output


def test_proof_metrics_require_assisted_and_unassisted_examples(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    store = Store(tmp_path / ".claude" / "mm.db", embedding_dims=4)
    for index in range(10):
        _task(store, f"assisted {index}", assisted=True, outcome=Outcome.SUCCESS, tokens=700)

    proof = compute_proof_metrics(store)

    assert proof.comparable_tasks == 10
    assert proof.assisted_tasks == 10
    assert proof.unassisted_tasks == 0
    assert proof.confidence is EvidenceConfidence.LOW
    assert proof.unassisted_success_rate is None
    assert proof.outcome_lift is None


def test_proof_metrics_high_confidence_requires_low_unknown_rate(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    store = Store(tmp_path / ".claude" / "mm.db", embedding_dims=4)
    for index in range(25):
        _task(store, f"assisted {index}", assisted=True, outcome=Outcome.SUCCESS, tokens=700)
    for index in range(15):
        _task(store, f"unassisted {index}", assisted=False, outcome=Outcome.SUCCESS, tokens=1000)
    for index in range(10):
        _task(store, f"unknown {index}", assisted=False, outcome=Outcome.UNKNOWN, tokens=1000)

    proof = compute_proof_metrics(store)

    assert proof.comparable_tasks == 50
    assert proof.token_samples == 50
    assert proof.unknown_outcomes == 10
    assert proof.confidence is EvidenceConfidence.HIGH


def test_proof_confidence_requires_known_paired_outcomes(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    store = Store(tmp_path / ".claude" / "mm.db", embedding_dims=4)
    for index in range(5):
        _task(store, f"assisted {index}", assisted=True, outcome=Outcome.UNKNOWN, tokens=700)
    for index in range(5):
        _task(
            store,
            f"unassisted {index}",
            assisted=False,
            outcome=Outcome.UNKNOWN,
            tokens=1000,
        )

    proof = compute_proof_metrics(store)

    assert proof.comparable_tasks == 10
    assert proof.assisted_success_rate is None
    assert proof.unassisted_success_rate is None
    assert proof.outcome_lift is None
    assert proof.confidence is EvidenceConfidence.LOW


def test_high_confidence_requires_paired_token_samples(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    store = Store(tmp_path / ".claude" / "mm.db", embedding_dims=4)
    for index in range(10):
        _task(store, f"assisted {index}", assisted=True, outcome=Outcome.SUCCESS, tokens=700)
    for index in range(40):
        _task(
            store,
            f"unassisted {index}",
            assisted=False,
            outcome=Outcome.FAILURE,
            tokens=None,
            output_tokens=None,
        )

    proof = compute_proof_metrics(store)

    assert proof.comparable_tasks == 50
    assert proof.token_reduction is None
    assert proof.confidence is EvidenceConfidence.MEDIUM


def test_token_reduction_requires_complete_token_samples(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    store = Store(tmp_path / ".claude" / "mm.db", embedding_dims=4)
    for index in range(25):
        _task(
            store,
            f"assisted {index}",
            assisted=True,
            outcome=Outcome.SUCCESS,
            tokens=700,
            output_tokens=None,
        )
    for index in range(25):
        _task(
            store,
            f"unassisted {index}",
            assisted=False,
            outcome=Outcome.FAILURE,
            tokens=1000,
            output_tokens=None,
        )

    proof = compute_proof_metrics(store)

    assert proof.comparable_tasks == 50
    assert proof.token_samples == 0
    assert proof.token_reduction is None
    assert proof.confidence is EvidenceConfidence.MEDIUM


def test_status_json_empty_store_has_stable_proof_contract(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    proof = json.loads(result.output)["proof"]
    assert proof == {
        "confidence": "low",
        "comparable_tasks": 0,
        "assisted_tasks": 0,
        "unassisted_tasks": 0,
        "assisted_success_rate": None,
        "unassisted_success_rate": None,
        "outcome_lift": None,
        "token_reduction": None,
        "token_samples": 0,
        "unknown_outcomes": 0,
    }
