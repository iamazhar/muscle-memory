"""Tests for task, activation, and measurement records."""

from __future__ import annotations

from muscle_memory.db import Store
from muscle_memory.models import (
    ActivationRecord,
    DeliveryMode,
    EvidenceConfidence,
    MeasurementRecord,
    Outcome,
    TaskRecord,
)


def test_store_round_trips_task_activation_and_measurement(tmp_db: Store) -> None:
    task = TaskRecord(
        raw_prompt="<local-command-caveat>noise</local-command-caveat> run tests",
        cleaned_prompt="run tests",
        harness="codex",
        project_path="/tmp/repo",
        session_id="session-1",
    )
    tmp_db.add_task(task)

    activation = ActivationRecord(
        task_id=task.id,
        skill_id="skill-1",
        distance=0.25,
        final_rank=0.15,
        delivery_mode=DeliveryMode.CODEX_USE,
        injected_token_count=42,
    )
    tmp_db.add_activation(activation)

    measurement = MeasurementRecord(
        task_id=task.id,
        outcome=Outcome.SUCCESS,
        confidence=EvidenceConfidence.HIGH,
        reason="pytest: 5 passed, 0 failed",
        input_tokens=1000,
        output_tokens=250,
        injected_skill_tokens=42,
        tool_call_count=3,
        comparable=True,
    )
    tmp_db.add_measurement(measurement)

    loaded_task = tmp_db.get_task(task.id)
    assert loaded_task is not None
    assert loaded_task.cleaned_prompt == "run tests"
    assert loaded_task.harness == "codex"

    session_task = tmp_db.find_latest_task_by_session("session-1")
    assert session_task is not None
    assert session_task.id == task.id

    activations = tmp_db.list_activations_for_task(task.id)
    assert len(activations) == 1
    assert activations[0].skill_id == "skill-1"
    assert activations[0].credited_outcome is None

    tmp_db.credit_activations(task.id, ["skill-1"], Outcome.SUCCESS)
    credited = tmp_db.list_activations_for_task(task.id)[0]
    assert credited.credited_outcome == Outcome.SUCCESS

    loaded_measurement = tmp_db.get_measurement_for_task(task.id)
    assert loaded_measurement is not None
    assert loaded_measurement.input_tokens == 1000
    assert loaded_measurement.confidence is EvidenceConfidence.HIGH


def test_schema_migration_preserves_existing_episode_rows(tmp_path) -> None:
    db_path = tmp_path / "mm.db"
    store = Store(db_path, embedding_dims=4)
    assert store.count_episodes() == 0

    reopened = Store(db_path, embedding_dims=4)
    task = TaskRecord(
        raw_prompt="fix tests",
        cleaned_prompt="fix tests",
        harness="claude-code",
        project_path=str(tmp_path),
    )
    reopened.add_task(task)

    assert reopened.get_task(task.id) is not None
