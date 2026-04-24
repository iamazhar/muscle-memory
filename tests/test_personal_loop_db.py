"""Tests for task, activation, and measurement records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from muscle_memory.db import Store
from muscle_memory.models import (
    ActivationRecord,
    DeliveryMode,
    EvidenceConfidence,
    MeasurementRecord,
    Outcome,
    TaskRecord,
    Trajectory,
)


def test_store_round_trips_task_activation_and_measurement(tmp_db: Store) -> None:
    older_task = TaskRecord(
        raw_prompt="older prompt",
        cleaned_prompt="older prompt",
        harness="codex",
        project_path="/tmp/repo",
        session_id="session-1",
        created_at=datetime(2026, 4, 23, 12, 0, tzinfo=UTC),
    )
    task = TaskRecord(
        raw_prompt="<local-command-caveat>noise</local-command-caveat> run tests",
        cleaned_prompt="run tests",
        harness="codex",
        project_path="/tmp/repo",
        session_id="session-1",
        created_at=older_task.created_at + timedelta(minutes=1),
    )
    tmp_db.add_task(older_task)
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


def test_measurement_upsert_preserves_original_row_identity(tmp_db: Store) -> None:
    task = TaskRecord(raw_prompt="run tests", cleaned_prompt="run tests")
    tmp_db.add_task(task)

    original = MeasurementRecord(
        id="measurement-original",
        task_id=task.id,
        outcome=Outcome.UNKNOWN,
        confidence=EvidenceConfidence.LOW,
        reason="initial evidence",
        input_tokens=100,
        output_tokens=20,
    )
    tmp_db.add_measurement(original)

    updated = MeasurementRecord(
        id="measurement-replacement",
        task_id=task.id,
        outcome=Outcome.SUCCESS,
        confidence=EvidenceConfidence.HIGH,
        reason="pytest: 5 passed",
        input_tokens=120,
        output_tokens=25,
        injected_skill_tokens=10,
        tool_call_count=3,
        comparable=True,
    )
    tmp_db.add_measurement(updated)

    loaded = tmp_db.get_measurement_for_task(task.id)
    assert loaded is not None
    assert loaded.id == original.id
    assert loaded.outcome is Outcome.SUCCESS
    assert loaded.confidence is EvidenceConfidence.HIGH
    assert loaded.reason == "pytest: 5 passed"
    assert loaded.input_tokens == 120
    assert loaded.output_tokens == 25
    assert loaded.injected_skill_tokens == 10
    assert loaded.tool_call_count == 3
    assert loaded.comparable is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Trajectory(input_tokens=-1),
        lambda: Trajectory(output_tokens=-1),
        lambda: ActivationRecord(
            task_id="task-1",
            skill_id="skill-1",
            delivery_mode=DeliveryMode.CODEX_USE,
            injected_token_count=-1,
        ),
        lambda: MeasurementRecord(task_id="task-1", input_tokens=-1),
        lambda: MeasurementRecord(task_id="task-1", output_tokens=-1),
        lambda: MeasurementRecord(task_id="task-1", injected_skill_tokens=-1),
        lambda: MeasurementRecord(task_id="task-1", tool_call_count=-1),
    ],
)
def test_record_models_reject_negative_token_and_count_fields(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_db_rejects_negative_token_and_count_fields(tmp_db: Store) -> None:
    task = TaskRecord(raw_prompt="run tests", cleaned_prompt="run tests")
    tmp_db.add_task(task)

    with pytest.raises(sqlite3.IntegrityError), tmp_db.batch() as conn:
        conn.execute(
            """
            INSERT INTO activations (
                id, task_id, skill_id, delivery_mode,
                injected_token_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "activation-negative",
                task.id,
                "skill-1",
                DeliveryMode.CODEX_USE.value,
                -1,
                datetime(2026, 4, 23, 12, 0, tzinfo=UTC).isoformat(),
            ),
        )

    with pytest.raises(sqlite3.IntegrityError), tmp_db.batch() as conn:
        conn.execute(
            """
            INSERT INTO measurements (
                id, task_id, outcome, confidence, reason,
                input_tokens, output_tokens, injected_skill_tokens,
                tool_call_count, comparable, measured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "measurement-negative",
                task.id,
                Outcome.SUCCESS.value,
                EvidenceConfidence.HIGH.value,
                "bad count",
                1,
                1,
                0,
                -1,
                1,
                datetime(2026, 4, 23, 12, 0, tzinfo=UTC).isoformat(),
            ),
        )


def test_schema_migration_preserves_existing_episode_rows(tmp_path) -> None:
    db_path = tmp_path / "mm.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_prompt TEXT NOT NULL,
                trajectory TEXT NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'unknown',
                reward REAL NOT NULL DEFAULT 0.0,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                project_path TEXT,
                activated_skills TEXT NOT NULL DEFAULT '[]'
            );
            """
        )
        conn.execute("INSERT INTO schema_version (version) VALUES (6)")
        conn.execute(
            """
            INSERT INTO episodes (
                id, session_id, user_prompt, trajectory, outcome, reward,
                started_at, ended_at, project_path, activated_skills
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "episode-1",
                "session-1",
                "fix tests",
                '{"user_prompt":"fix tests","tool_calls":[],"assistant_turns":[]}',
                Outcome.SUCCESS.value,
                0.8,
                datetime(2026, 4, 23, 12, 0, tzinfo=UTC).isoformat(),
                None,
                str(tmp_path),
                "[]",
            ),
        )

    reopened = Store(db_path, embedding_dims=4)
    assert reopened.count_episodes() == 1
    episode = reopened.get_episode("episode-1")
    assert episode is not None
    assert episode.user_prompt == "fix tests"
    assert episode.outcome is Outcome.SUCCESS

    task = TaskRecord(
        raw_prompt="fix tests",
        cleaned_prompt="fix tests",
        harness="claude-code",
        project_path=str(tmp_path),
    )
    reopened.add_task(task)

    assert reopened.get_task(task.id) is not None
    with sqlite3.connect(db_path) as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'index' AND tbl_name IN ('tasks', 'measurements')
                """
            )
        }
    assert "idx_tasks_session_created" in indexes
    assert "idx_measurements_measured_at" in indexes


def test_schema_migration_adds_constraints_to_version_7_personal_loop_tables(
    tmp_path,
) -> None:
    db_path = tmp_path / "mm.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                raw_prompt TEXT NOT NULL,
                cleaned_prompt TEXT NOT NULL,
                intent_summary TEXT,
                harness TEXT NOT NULL DEFAULT 'generic',
                project_path TEXT,
                session_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE activations (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                skill_id TEXT NOT NULL,
                distance REAL,
                final_rank REAL,
                delivery_mode TEXT NOT NULL,
                injected_token_count INTEGER NOT NULL DEFAULT 0,
                credited_outcome TEXT,
                created_at TEXT NOT NULL,
                credited_at TEXT
            );

            CREATE TABLE measurements (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE REFERENCES tasks(id) ON DELETE CASCADE,
                outcome TEXT NOT NULL DEFAULT 'unknown',
                confidence TEXT NOT NULL DEFAULT 'low',
                reason TEXT NOT NULL DEFAULT '',
                input_tokens INTEGER,
                output_tokens INTEGER,
                injected_skill_tokens INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                comparable INTEGER NOT NULL DEFAULT 0,
                measured_at TEXT NOT NULL
            );
            """
        )
        conn.execute("INSERT INTO schema_version (version) VALUES (7)")
        conn.execute(
            """
            INSERT INTO tasks (
                id, raw_prompt, cleaned_prompt, harness, session_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "task-1",
                "run tests",
                "run tests",
                "codex",
                "session-1",
                datetime(2026, 4, 23, 12, 0, tzinfo=UTC).isoformat(),
            ),
        )
        conn.execute(
            """
            INSERT INTO activations (
                id, task_id, skill_id, delivery_mode,
                injected_token_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "activation-1",
                "task-1",
                "skill-1",
                DeliveryMode.CODEX_USE.value,
                5,
                datetime(2026, 4, 23, 12, 1, tzinfo=UTC).isoformat(),
            ),
        )
        conn.execute(
            """
            INSERT INTO measurements (
                id, task_id, outcome, confidence, reason, input_tokens,
                output_tokens, injected_skill_tokens, tool_call_count,
                comparable, measured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "measurement-1",
                "task-1",
                Outcome.SUCCESS.value,
                EvidenceConfidence.HIGH.value,
                "pytest passed",
                100,
                25,
                5,
                3,
                1,
                datetime(2026, 4, 23, 12, 2, tzinfo=UTC).isoformat(),
            ),
        )

    reopened = Store(db_path, embedding_dims=4)
    assert reopened.list_activations_for_task("task-1")[0].id == "activation-1"
    measurement = reopened.get_measurement_for_task("task-1")
    assert measurement is not None
    assert measurement.id == "measurement-1"
    assert measurement.input_tokens == 100

    with pytest.raises(sqlite3.IntegrityError), reopened.batch() as conn:
        conn.execute(
            """
            INSERT INTO measurements (
                id, task_id, outcome, confidence, reason,
                input_tokens, output_tokens, injected_skill_tokens,
                tool_call_count, comparable, measured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "measurement-negative",
                "task-1",
                Outcome.SUCCESS.value,
                EvidenceConfidence.HIGH.value,
                "bad tokens",
                -1,
                1,
                0,
                0,
                1,
                datetime(2026, 4, 23, 12, 3, tzinfo=UTC).isoformat(),
            ),
        )
