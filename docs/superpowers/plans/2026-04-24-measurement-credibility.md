# Measurement Credibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first personal compounding loop milestone: real task capture, canonical activation records, Codex-friendly `mm use`, token/outcome measurements, and proof-oriented `mm status` output.

**Architecture:** Add first-class `TaskRecord`, `ActivationRecord`, and `MeasurementRecord` models backed by SQLite tables. Route both Claude Code hook retrieval and manual Codex-style `mm use` through the same capture and activation path, then attach stop/ingest outcomes and token evidence to those records. Keep existing episode/skill behavior compatible while adding proof metrics to `mm status`.

**Tech Stack:** Python 3.12, Typer, Rich, Pydantic, SQLite, pytest, ruff, mypy.

---

## Scope

This plan implements Milestone 1 from `docs/superpowers/specs/2026-04-23-personal-compounding-loop-design.md`.

In scope:

- `tasks`, `activations`, and `measurements` tables.
- Pydantic models and Store DAO methods for those tables.
- `mm use "<task>"` as the documented manual/Codex path.
- Claude Code hook writes to the canonical activation table.
- Stop hook and transcript ingest write measurements.
- Codex transcript token usage is captured when present.
- `mm status` gains a proof section with confidence-labeled outcome/token evidence.
- `mm retrieve` remains available as a hidden compatibility alias.

Out of scope:

- Team sharing.
- Cloud sync.
- Demo rewrite.
- README rewrite beyond the minimum CLI/help adjustment.
- Codex automatic prompt hooks.
- Removing advanced internal commands.
- Claiming token savings when token evidence is absent.

## File Structure

- Modify `src/muscle_memory/models.py`: add record models and lightweight enums for delivery mode and confidence.
- Modify `src/muscle_memory/db.py`: bump schema version, create/migrate tables, add DAO methods.
- Create `src/muscle_memory/personal_loop.py`: task capture, activation recording, token counting, proof computation, and `mm use` context formatting.
- Modify `src/muscle_memory/cli.py`: add `mm use`, hide `retrieve`, and render proof metrics in `mm status`.
- Modify `src/muscle_memory/hooks/user_prompt.py`: route Claude Code hook retrieval through task/activation records while preserving hook safety.
- Modify `src/muscle_memory/hooks/stop.py`: load canonical activations, credit them, and write measurements; fall back to sidecar activations for older sessions.
- Modify `src/muscle_memory/harness/codex.py`: capture `turn.completed.usage.input_tokens` and `output_tokens` in the returned trajectory.
- Modify `src/muscle_memory/harness/claude_code.py`: capture usage fields if present in transcript records.
- Modify `src/muscle_memory/ingest.py`: create task and measurement records for learned transcripts.
- Modify `tests/test_cli_help.py`: public help shows `use` and no longer shows `retrieve`.
- Add `tests/test_personal_loop_db.py`: schema and DAO coverage.
- Add `tests/test_cli_use.py`: CLI behavior and activation recording.
- Add `tests/test_hooks_personal_loop.py`: Claude hook/stop attribution.
- Add `tests/test_ingest_measurement.py`: transcript token measurement.
- Add `tests/test_status_proof.py`: proof dashboard and JSON metrics.

---

### Task 1: Data Models And SQLite Tables

**Files:**

- Modify: `src/muscle_memory/models.py`
- Modify: `src/muscle_memory/db.py`
- Create: `tests/test_personal_loop_db.py`

- [ ] **Step 1: Write failing DAO tests**

Create `tests/test_personal_loop_db.py`:

```python
"""Tests for task, activation, and measurement records."""

from __future__ import annotations

from datetime import UTC, datetime

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
```

- [ ] **Step 2: Run DAO tests and verify they fail**

Run:

```bash
uv run pytest tests/test_personal_loop_db.py -q
```

Expected: FAIL with an import error for `TaskRecord`, `ActivationRecord`, `MeasurementRecord`, `DeliveryMode`, or `EvidenceConfidence`.

- [ ] **Step 3: Add model classes**

In `src/muscle_memory/models.py`, add these enums near the existing enums:

```python
class DeliveryMode(str, Enum):
    """How a skill was delivered to an agent."""

    CLAUDE_HOOK = "claude-hook"
    CODEX_USE = "codex-use"
    MANUAL = "manual"


class EvidenceConfidence(str, Enum):
    """Confidence level for outcome and token measurements."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
```

Extend `Trajectory` with optional token fields:

```python
class Trajectory(BaseModel):
    """A sequence of turns and tool calls during a task.

    Represented flatly as an ordered list of events so that the
    extractor can reason about call order and causality.
    """

    user_prompt: str = ""
    user_followup: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    assistant_turns: list[str] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
```

Add these record models after `Episode`:

```python
class TaskRecord(BaseModel):
    """A captured user task before or during skill retrieval."""

    id: str = Field(default_factory=_new_id)
    raw_prompt: str
    cleaned_prompt: str
    intent_summary: str | None = None
    harness: str = "generic"
    project_path: str | None = None
    session_id: str | None = None
    created_at: datetime = Field(default_factory=_now)


class ActivationRecord(BaseModel):
    """A skill offered or injected for a captured task."""

    id: str = Field(default_factory=_new_id)
    task_id: str
    skill_id: str
    distance: float | None = None
    final_rank: float | None = None
    delivery_mode: DeliveryMode
    injected_token_count: int = 0
    credited_outcome: Outcome | None = None
    created_at: datetime = Field(default_factory=_now)
    credited_at: datetime | None = None


class MeasurementRecord(BaseModel):
    """Outcome and token evidence for a captured task."""

    id: str = Field(default_factory=_new_id)
    task_id: str
    outcome: Outcome = Outcome.UNKNOWN
    confidence: EvidenceConfidence = EvidenceConfidence.LOW
    reason: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    injected_skill_tokens: int = 0
    tool_call_count: int = 0
    comparable: bool = False
    measured_at: datetime = Field(default_factory=_now)
```

- [ ] **Step 4: Add SQLite schema and migrations**

In `src/muscle_memory/db.py`, update imports:

```python
from muscle_memory.models import (
    ActivationRecord,
    BackgroundJob,
    DeliveryMode,
    Episode,
    EvidenceConfidence,
    JobKind,
    JobStatus,
    Maturity,
    MeasurementRecord,
    Outcome,
    Scope,
    Skill,
    TaskRecord,
    ToolCall,
    Trajectory,
)
```

Change:

```python
SCHEMA_VERSION = 7
```

Inside `_init_db()` after the `jobs` table block, add:

```python
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    raw_prompt TEXT NOT NULL,
                    cleaned_prompt TEXT NOT NULL,
                    intent_summary TEXT,
                    harness TEXT NOT NULL DEFAULT 'generic',
                    project_path TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);

                CREATE TABLE IF NOT EXISTS activations (
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

                CREATE INDEX IF NOT EXISTS idx_activations_task ON activations(task_id);
                CREATE INDEX IF NOT EXISTS idx_activations_skill ON activations(skill_id);

                CREATE TABLE IF NOT EXISTS measurements (
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

                CREATE INDEX IF NOT EXISTS idx_measurements_outcome ON measurements(outcome);
                CREATE INDEX IF NOT EXISTS idx_measurements_comparable ON measurements(comparable);
```

At the end of `_migrate()` before the schema version update, add:

```python
        if current_version < 7:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    raw_prompt TEXT NOT NULL,
                    cleaned_prompt TEXT NOT NULL,
                    intent_summary TEXT,
                    harness TEXT NOT NULL DEFAULT 'generic',
                    project_path TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);

                CREATE TABLE IF NOT EXISTS activations (
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
                CREATE INDEX IF NOT EXISTS idx_activations_task ON activations(task_id);
                CREATE INDEX IF NOT EXISTS idx_activations_skill ON activations(skill_id);

                CREATE TABLE IF NOT EXISTS measurements (
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
                CREATE INDEX IF NOT EXISTS idx_measurements_outcome ON measurements(outcome);
                CREATE INDEX IF NOT EXISTS idx_measurements_comparable ON measurements(comparable);
                """
            )
```

- [ ] **Step 5: Add DAO methods and row mappers**

In `src/muscle_memory/db.py`, add these methods after `count_episodes()`:

```python
    def add_task(self, task: TaskRecord) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, raw_prompt, cleaned_prompt, intent_summary, harness,
                    project_path, session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.raw_prompt,
                    task.cleaned_prompt,
                    task.intent_summary,
                    task.harness,
                    task.project_path,
                    task.session_id,
                    _iso(task.created_at),
                ),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return _row_to_task(row) if row else None
        finally:
            conn.close()

    def find_latest_task_by_session(self, session_id: str) -> TaskRecord | None:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return _row_to_task(row) if row else None
        finally:
            conn.close()

    def list_tasks(self, *, limit: int | None = 50) -> list[TaskRecord]:
        conn = self._open()
        try:
            if limit is None:
                rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [_row_to_task(row) for row in rows]
        finally:
            conn.close()

    def add_activation(self, activation: ActivationRecord) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                INSERT INTO activations (
                    id, task_id, skill_id, distance, final_rank, delivery_mode,
                    injected_token_count, credited_outcome, created_at, credited_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    activation.id,
                    activation.task_id,
                    activation.skill_id,
                    activation.distance,
                    activation.final_rank,
                    activation.delivery_mode.value,
                    activation.injected_token_count,
                    activation.credited_outcome.value
                    if activation.credited_outcome is not None
                    else None,
                    _iso(activation.created_at),
                    _iso(activation.credited_at),
                ),
            )

    def list_activations_for_task(self, task_id: str) -> list[ActivationRecord]:
        conn = self._open()
        try:
            rows = conn.execute(
                "SELECT * FROM activations WHERE task_id = ? ORDER BY created_at ASC",
                (task_id,),
            ).fetchall()
            return [_row_to_activation(row) for row in rows]
        finally:
            conn.close()

    def credit_activations(
        self,
        task_id: str,
        skill_ids: list[str],
        outcome: Outcome,
    ) -> None:
        if not skill_ids:
            return
        now = _iso(datetime.now(UTC))
        skill_slots = ",".join("?" * len(skill_ids))
        with self.batch() as conn:
            conn.execute(
                f"""
                UPDATE activations
                   SET credited_outcome = ?, credited_at = ?
                 WHERE task_id = ? AND skill_id IN ({skill_slots})
                """,
                [outcome.value, now, task_id, *skill_ids],
            )

    def add_measurement(self, measurement: MeasurementRecord) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO measurements (
                    id, task_id, outcome, confidence, reason, input_tokens,
                    output_tokens, injected_skill_tokens, tool_call_count,
                    comparable, measured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    measurement.id,
                    measurement.task_id,
                    measurement.outcome.value,
                    measurement.confidence.value,
                    measurement.reason,
                    measurement.input_tokens,
                    measurement.output_tokens,
                    measurement.injected_skill_tokens,
                    measurement.tool_call_count,
                    1 if measurement.comparable else 0,
                    _iso(measurement.measured_at),
                ),
            )

    def get_measurement_for_task(self, task_id: str) -> MeasurementRecord | None:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM measurements WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return _row_to_measurement(row) if row else None
        finally:
            conn.close()

    def list_measurements(self, *, limit: int | None = 1000) -> list[MeasurementRecord]:
        conn = self._open()
        try:
            if limit is None:
                rows = conn.execute(
                    "SELECT * FROM measurements ORDER BY measured_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM measurements ORDER BY measured_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [_row_to_measurement(row) for row in rows]
        finally:
            conn.close()
```

Add these row mappers near the existing `_row_to_*` helpers:

```python
def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        raw_prompt=row["raw_prompt"],
        cleaned_prompt=row["cleaned_prompt"],
        intent_summary=row["intent_summary"],
        harness=row["harness"],
        project_path=row["project_path"],
        session_id=row["session_id"],
        created_at=_parse_iso(row["created_at"]) or datetime.now(UTC),
    )


def _row_to_activation(row: sqlite3.Row) -> ActivationRecord:
    credited = row["credited_outcome"]
    return ActivationRecord(
        id=row["id"],
        task_id=row["task_id"],
        skill_id=row["skill_id"],
        distance=row["distance"],
        final_rank=row["final_rank"],
        delivery_mode=DeliveryMode(row["delivery_mode"]),
        injected_token_count=int(row["injected_token_count"]),
        credited_outcome=Outcome(credited) if credited else None,
        created_at=_parse_iso(row["created_at"]) or datetime.now(UTC),
        credited_at=_parse_iso(row["credited_at"]),
    )


def _row_to_measurement(row: sqlite3.Row) -> MeasurementRecord:
    return MeasurementRecord(
        id=row["id"],
        task_id=row["task_id"],
        outcome=Outcome(row["outcome"]),
        confidence=EvidenceConfidence(row["confidence"]),
        reason=row["reason"] or "",
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        injected_skill_tokens=int(row["injected_skill_tokens"]),
        tool_call_count=int(row["tool_call_count"]),
        comparable=bool(row["comparable"]),
        measured_at=_parse_iso(row["measured_at"]) or datetime.now(UTC),
    )
```

- [ ] **Step 6: Run DAO tests and commit**

Run:

```bash
uv run pytest tests/test_personal_loop_db.py tests/test_db.py -q
```

Expected: PASS.

Commit:

```bash
git add src/muscle_memory/models.py src/muscle_memory/db.py tests/test_personal_loop_db.py
git commit -m "feat: add personal loop records"
```

---

### Task 2: Personal Loop Helpers And `mm use`

**Files:**

- Create: `src/muscle_memory/personal_loop.py`
- Modify: `src/muscle_memory/cli.py`
- Modify: `tests/test_cli_help.py`
- Create: `tests/test_cli_use.py`

- [ ] **Step 1: Write failing CLI tests for `mm use`**

Create `tests/test_cli_use.py`:

```python
"""Tests for the mm use command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import DeliveryMode, Maturity, Scope, Skill

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
    assert payload["context_token_count"] > 0


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
```

Modify `tests/test_cli_help.py`:

```python
def test_top_level_help_is_trimmed() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    out = result.output
    assert "init" in out
    assert "learn" in out
    assert "use" in out
    assert "skills" in out
    assert "show" in out
    assert "status" in out
    assert "doctor" in out

    assert "│ retrieve" not in out
    assert "│ list" not in out
    assert "│ stats" not in out
    assert "│ bootstrap" not in out
    assert "│ refine" not in out
    assert "│ ingest" not in out
    assert "│ maint" not in out
    assert "│ share" not in out
    assert "│ review" not in out
    assert "│ jobs" not in out
    assert "│ eval" not in out
    assert "│ simulate" not in out
    assert "│ hook" not in out
    assert "│ version" not in out
    assert "│ pause" not in out
    assert "│ resume" not in out
    assert "│ dedup" not in out
    assert "│ rescore" not in out
    assert "│ prune" not in out
    assert "│ export" not in out
    assert "│ import" not in out
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run pytest tests/test_cli_use.py tests/test_cli_help.py -q
```

Expected: FAIL because `mm use` is not registered and `retrieve` is still public.

- [ ] **Step 3: Implement personal loop helper module**

Create `src/muscle_memory/personal_loop.py`:

```python
"""Personal compounding loop helpers."""

from __future__ import annotations

from dataclasses import dataclass

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import (
    ActivationRecord,
    DeliveryMode,
    EvidenceConfidence,
    MeasurementRecord,
    Outcome,
    TaskRecord,
)
from muscle_memory.prompt_cleaning import clean_prompt_text
from muscle_memory.retriever import RetrievedSkill


def count_text_tokens(text: str) -> int:
    """Cheap deterministic token estimate for locally formatted context."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def capture_task(
    store: Store,
    *,
    raw_prompt: str,
    harness: str,
    project_path: str | None,
    session_id: str | None = None,
    cleaned_prompt: str | None = None,
) -> TaskRecord:
    cleaned = cleaned_prompt if cleaned_prompt is not None else clean_prompt_text(raw_prompt)
    task = TaskRecord(
        raw_prompt=raw_prompt,
        cleaned_prompt=cleaned or raw_prompt.strip() or "(unknown)",
        harness=harness,
        project_path=project_path,
        session_id=session_id,
    )
    store.add_task(task)
    return task


def format_context(hits: list[RetrievedSkill]) -> str:
    """Format playbooks for manual/Codex use."""
    if not hits:
        return ""
    lines = [
        "<muscle_memory>",
        "Use the relevant playbook below only if its activation matches the task.",
        "If it matches, execute the steps directly: run commands, edit files, and verify.",
        "If it does not match, ignore it and proceed normally.",
        "",
    ]
    for index, hit in enumerate(hits, start=1):
        skill = hit.skill
        lines.extend(
            [
                f"## Playbook {index}",
                f"Activate when: {skill.activation}",
                "Steps:",
                skill.execution,
                f"Done when: {skill.termination}",
            ]
        )
        if skill.tool_hints:
            lines.append(f"Preferred tools: {', '.join(skill.tool_hints)}")
        lines.append("")
    lines.append("</muscle_memory>")
    return "\n".join(lines)


def record_activations(
    store: Store,
    *,
    task: TaskRecord,
    hits: list[RetrievedSkill],
    delivery_mode: DeliveryMode,
    context_token_count: int,
) -> list[ActivationRecord]:
    if not hits:
        return []
    per_hit_tokens = max(1, context_token_count // len(hits))
    records: list[ActivationRecord] = []
    for hit in hits:
        record = ActivationRecord(
            task_id=task.id,
            skill_id=hit.skill.id,
            distance=hit.distance,
            final_rank=hit.final_rank,
            delivery_mode=delivery_mode,
            injected_token_count=per_hit_tokens,
        )
        store.add_activation(record)
        records.append(record)
    return records


def measurement_confidence(
    *,
    outcome: Outcome,
    input_tokens: int | None,
    output_tokens: int | None,
) -> EvidenceConfidence:
    if outcome is Outcome.UNKNOWN:
        return EvidenceConfidence.LOW
    if input_tokens is not None or output_tokens is not None:
        return EvidenceConfidence.HIGH
    return EvidenceConfidence.MEDIUM


def add_measurement_for_task(
    store: Store,
    *,
    task: TaskRecord,
    outcome: Outcome,
    reason: str,
    input_tokens: int | None,
    output_tokens: int | None,
    injected_skill_tokens: int,
    tool_call_count: int,
    comparable: bool,
) -> MeasurementRecord:
    measurement = MeasurementRecord(
        task_id=task.id,
        outcome=outcome,
        confidence=measurement_confidence(
            outcome=outcome,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        reason=reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        injected_skill_tokens=injected_skill_tokens,
        tool_call_count=tool_call_count,
        comparable=comparable,
    )
    store.add_measurement(measurement)
    return measurement


@dataclass
class ProofMetrics:
    confidence: EvidenceConfidence
    comparable_tasks: int
    assisted_tasks: int
    unassisted_tasks: int
    assisted_success_rate: float | None
    unassisted_success_rate: float | None
    outcome_lift: float | None
    token_reduction: float | None
    token_samples: int
    unknown_outcomes: int


def compute_proof_metrics(store: Store) -> ProofMetrics:
    measurements = [m for m in store.list_measurements(limit=10_000) if m.comparable]
    task_ids_with_activation = {
        task.id
        for task in store.list_tasks(limit=None)
        if store.list_activations_for_task(task.id)
    }
    assisted = [m for m in measurements if m.task_id in task_ids_with_activation]
    unassisted = [m for m in measurements if m.task_id not in task_ids_with_activation]
    assisted_known = [m for m in assisted if m.outcome is not Outcome.UNKNOWN]
    unassisted_known = [m for m in unassisted if m.outcome is not Outcome.UNKNOWN]

    assisted_success_rate = _success_rate(assisted_known)
    unassisted_success_rate = _success_rate(unassisted_known)
    outcome_lift = (
        assisted_success_rate - unassisted_success_rate
        if assisted_success_rate is not None and unassisted_success_rate is not None
        else None
    )

    assisted_tokens = [m.input_tokens for m in assisted if m.input_tokens is not None]
    unassisted_tokens = [m.input_tokens for m in unassisted if m.input_tokens is not None]
    token_reduction = None
    if assisted_tokens and unassisted_tokens:
        assisted_avg = sum(assisted_tokens) / len(assisted_tokens)
        unassisted_avg = sum(unassisted_tokens) / len(unassisted_tokens)
        if unassisted_avg > 0:
            token_reduction = (unassisted_avg - assisted_avg) / unassisted_avg

    token_samples = len(assisted_tokens) + len(unassisted_tokens)
    unknowns = sum(1 for m in measurements if m.outcome is Outcome.UNKNOWN)
    confidence = _proof_confidence(
        comparable_count=len(measurements),
        assisted_count=len(assisted),
        unassisted_count=len(unassisted),
        token_samples=token_samples,
        unknowns=unknowns,
    )
    return ProofMetrics(
        confidence=confidence,
        comparable_tasks=len(measurements),
        assisted_tasks=len(assisted),
        unassisted_tasks=len(unassisted),
        assisted_success_rate=assisted_success_rate,
        unassisted_success_rate=unassisted_success_rate,
        outcome_lift=outcome_lift,
        token_reduction=token_reduction,
        token_samples=token_samples,
        unknown_outcomes=unknowns,
    )


def _success_rate(measurements: list[MeasurementRecord]) -> float | None:
    if not measurements:
        return None
    successes = sum(1 for m in measurements if m.outcome is Outcome.SUCCESS)
    return successes / len(measurements)


def _proof_confidence(
    *,
    comparable_count: int,
    assisted_count: int,
    unassisted_count: int,
    token_samples: int,
    unknowns: int,
) -> EvidenceConfidence:
    if comparable_count >= 50 and assisted_count >= 10 and unassisted_count >= 10 and token_samples >= 20 and unknowns / comparable_count <= 0.2:
        return EvidenceConfidence.HIGH
    if comparable_count >= 10 and assisted_count >= 3 and unassisted_count >= 3:
        return EvidenceConfidence.MEDIUM
    return EvidenceConfidence.LOW
```

- [ ] **Step 4: Register `mm use` and hide `retrieve`**

In `src/muscle_memory/cli.py`, change the retrieve decorator:

```python
@app.command(hidden=True)
def retrieve(
```

Add a public `use` command immediately before `retrieve`:

```python
@app.command("use")
def use_skill(
    prompt: str = typer.Argument(..., help="Task to get practiced context for."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Use practiced skill for a task."""
    from muscle_memory.personal_loop import (
        capture_task,
        count_text_tokens,
        format_context,
        record_activations,
    )
    from muscle_memory.retriever import Retriever

    cfg = _load_config()
    store = _open_store(cfg)
    task = capture_task(
        store,
        raw_prompt=prompt,
        harness=cfg.harness,
        project_path=str(cfg.project_root) if cfg.project_root is not None else None,
    )
    embedder = make_embedder(cfg)
    hits = Retriever(store, embedder, cfg).retrieve(task.cleaned_prompt)
    context = format_context(hits)
    context_token_count = count_text_tokens(context)
    activations = record_activations(
        store,
        task=task,
        hits=hits,
        delivery_mode=DeliveryMode.CODEX_USE if cfg.harness == "codex" else DeliveryMode.MANUAL,
        context_token_count=context_token_count,
    )

    if as_json:
        payload = {
            "task": {
                "id": task.id,
                "cleaned_prompt": task.cleaned_prompt,
                "harness": task.harness,
            },
            "context": context,
            "context_token_count": context_token_count,
            "hits": [
                {
                    **_skill_to_dict(hit.skill),
                    "distance": hit.distance,
                    "final_rank": hit.final_rank,
                    "activation_id": activations[index].id,
                }
                for index, hit in enumerate(hits)
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not hits:
        console.print("[dim]No matching skills. Proceed normally.[/dim]")
        return
    console.print(context)
```

Add `DeliveryMode` to the `models` import at the top of `cli.py`:

```python
from muscle_memory.models import BackgroundJob, DeliveryMode, JobKind, JobStatus, Maturity, Outcome, Scope, Skill
```

- [ ] **Step 5: Run CLI tests and commit**

Run:

```bash
uv run pytest tests/test_cli_use.py tests/test_cli_help.py tests/test_cli_harness.py::test_retrieve_returns_matching_skills_as_json -q
```

Expected: PASS. The retrieve compatibility test must still pass even though `retrieve` is hidden from top-level help.

Commit:

```bash
git add src/muscle_memory/personal_loop.py src/muscle_memory/cli.py tests/test_cli_use.py tests/test_cli_help.py
git commit -m "feat: add mm use workflow"
```

---

### Task 3: Route Claude Hook Retrieval Through Canonical Activations

**Files:**

- Modify: `src/muscle_memory/hooks/user_prompt.py`
- Modify: `src/muscle_memory/hooks/stop.py`
- Create: `tests/test_hooks_personal_loop.py`

- [ ] **Step 1: Write failing hook tests**

Create `tests/test_hooks_personal_loop.py`:

```python
"""Tests for hook integration with task and activation records."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.hooks import stop, user_prompt
from muscle_memory.models import DeliveryMode, Maturity, Outcome, Scope, Skill


class DummyEmbedder:
    dims = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        tokens = text.lower()
        return [
            1.0 if "pytest" in tokens else 0.0,
            1.0 if "import" in tokens else 0.0,
            1.0 if "runner" in tokens else 0.0,
            0.0,
        ]


def _config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
        harness="claude-code",
        auto_refine_enabled=False,
    )


def _seed_skill(store: Store, embedder: DummyEmbedder) -> Skill:
    skill = Skill(
        activation="When pytest fails with import errors",
        execution="1. Use the repo test runner\n2. Rerun pytest",
        termination="Tests pass",
        maturity=Maturity.LIVE,
        successes=2,
        invocations=2,
        score=1.0,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill, embedding=embedder.embed_one(skill.activation))
    return skill


def test_user_prompt_hook_records_task_and_activation(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "prompt": "pytest import errors",
    }

    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.user_prompt.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.user_prompt.make_embedder", return_value=embedder),
    ):
        exit_code = user_prompt.main([])

    assert exit_code == 0
    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    assert tasks[0].session_id == "session-1"
    activations = store.list_activations_for_task(tasks[0].id)
    assert len(activations) == 1
    assert activations[0].skill_id == skill.id
    assert activations[0].delivery_mode is DeliveryMode.CLAUDE_HOOK


def test_stop_hook_credits_canonical_activation_and_records_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    from muscle_memory.personal_loop import capture_task, record_activations
    from muscle_memory.retriever import RetrievedSkill

    task = capture_task(
        store,
        raw_prompt="pytest import errors",
        cleaned_prompt="pytest import errors",
        harness="claude-code",
        project_path=str(project_root),
        session_id="session-1",
    )
    record_activations(
        store,
        task=task,
        hits=[RetrievedSkill(skill=skill, distance=0.2, score_bonus=0.1)],
        delivery_mode=DeliveryMode.CLAUDE_HOOK,
        context_token_count=40,
    )

    transcript = project_root / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "pytest import errors"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": "pytest"},
                                }
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
                                    "content": "5 passed in 1.0s",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "transcript_path": str(transcript),
    }

    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.stop._fire_async_extraction"),
    ):
        exit_code = stop.main([])

    assert exit_code == 0
    credited = store.list_activations_for_task(task.id)[0]
    assert credited.credited_outcome is Outcome.SUCCESS
    measurement = store.get_measurement_for_task(task.id)
    assert measurement is not None
    assert measurement.outcome is Outcome.SUCCESS
    assert measurement.injected_skill_tokens == 40


def _stdin(payload: dict):
    class Stdin:
        def read(self) -> str:
            return json.dumps(payload)
    return Stdin()
```

- [ ] **Step 2: Run hook tests and verify they fail**

Run:

```bash
uv run pytest tests/test_hooks_personal_loop.py -q
```

Expected: FAIL because hooks still write only sidecar activation records.

- [ ] **Step 3: Update user prompt hook**

In `src/muscle_memory/hooks/user_prompt.py`, import helpers:

```python
from muscle_memory.models import DeliveryMode
from muscle_memory.personal_loop import (
    capture_task,
    count_text_tokens,
    record_activations,
)
```

After `store = Store(...)` and before retrieval, create a task:

```python
        task = capture_task(
            store,
            raw_prompt=prompt,
            harness=cfg.harness,
            project_path=cwd,
            session_id=session_id,
        )
```

Change retrieval to use the cleaned prompt:

```python
        hits = retriever.retrieve(task.cleaned_prompt)
```

After context creation, record canonical activations. Replace the existing activation write block with this shape:

```python
    context = adapter.format_context(hits)
    try:
        activation_start = time.perf_counter()
        new_hits = [h for h in hits if h.skill.id not in existing_activation_ids]
        if new_hits:
            retriever.mark_activated(new_hits)
        _record_activation(
            cfg,
            session_id,
            [{"skill_id": h.skill.id, "distance": h.distance} for h in hits],
        )
        activation_records = record_activations(
            store,
            task=task,
            hits=hits,
            delivery_mode=DeliveryMode.CLAUDE_HOOK,
            context_token_count=count_text_tokens(context),
        )
        activation_record_ms = (time.perf_counter() - activation_start) * 1000
        log_debug_event(
            cfg,
            component="user_prompt",
            event="hits_returned",
            session_id=session_id,
            task_id=task.id,
            activation_ids=[record.id for record in activation_records],
            prompt_excerpt=task.cleaned_prompt[:120],
            hit_count=len(hits),
            new_hit_count=len(new_hits),
            skill_ids=[h.skill.id for h in hits],
            distances=[round(h.distance, 4) for h in hits],
            retrieve_ms=round(retrieve_ms, 3),
            embed_ms=round(retriever.last_diagnostics.embed_ms, 3),
            search_ms=round(retriever.last_diagnostics.search_ms, 3),
            rerank_ms=round(retriever.last_diagnostics.rerank_ms, 3),
            activation_record_ms=round(activation_record_ms, 3),
            total_ms=round(retrieve_ms + activation_record_ms, 3),
            candidate_hits=retriever.last_diagnostics.candidate_hits,
            final_hits=retriever.last_diagnostics.final_hits,
            lexical_prefilter_skipped=retriever.last_diagnostics.lexical_prefilter_skipped,
        )
    except Exception:
        pass
```

Keep `_record_activation` and `_load_recorded_activation_ids` for backward compatibility.

- [ ] **Step 4: Update stop hook measurement path**

In `src/muscle_memory/hooks/stop.py`, add imports:

```python
from muscle_memory.personal_loop import add_measurement_for_task
```

After `store = Store(...)`, load the task:

```python
        task = store.find_latest_task_by_session(session_id) if session_id else None
```

Replace activation loading with canonical-first behavior:

```python
        if task is not None:
            activation_records = store.list_activations_for_task(task.id)
            activated = [record.skill_id for record in activation_records]
        else:
            activation_records = []
            activated = _load_activations(cfg, session_id)
```

After `Scorer(...).credit_episode(episode)`, add:

```python
        if task is not None:
            store.credit_activations(task.id, activated, signal.outcome)
            injected_tokens = sum(record.injected_token_count for record in activation_records)
            add_measurement_for_task(
                store,
                task=task,
                outcome=signal.outcome,
                reason="; ".join(signal.reasons),
                input_tokens=trajectory.input_tokens,
                output_tokens=trajectory.output_tokens,
                injected_skill_tokens=injected_tokens,
                tool_call_count=len(trajectory.tool_calls),
                comparable=bool(trajectory.tool_calls),
            )
```

- [ ] **Step 5: Run hook tests and commit**

Run:

```bash
uv run pytest tests/test_hooks_personal_loop.py tests/test_edge_cases.py::TestHookFailures -q
```

Expected: PASS. If `TestHookFailures` is not a class in the current file, run:

```bash
uv run pytest tests/test_hooks_personal_loop.py tests/test_edge_cases.py -q
```

Commit:

```bash
git add src/muscle_memory/hooks/user_prompt.py src/muscle_memory/hooks/stop.py tests/test_hooks_personal_loop.py
git commit -m "feat: record canonical hook activations"
```

---

### Task 4: Transcript Token Measurement

**Files:**

- Modify: `src/muscle_memory/harness/codex.py`
- Modify: `src/muscle_memory/harness/claude_code.py`
- Modify: `src/muscle_memory/ingest.py`
- Create: `tests/test_ingest_measurement.py`

- [ ] **Step 1: Write failing ingest measurement tests**

Create `tests/test_ingest_measurement.py`:

```python
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
```

- [ ] **Step 2: Run ingest measurement tests and verify they fail**

Run:

```bash
uv run pytest tests/test_ingest_measurement.py -q
```

Expected: FAIL because transcript usage is not stored on `Trajectory` and ingest does not create task/measurement records.

- [ ] **Step 3: Capture Codex usage**

In `src/muscle_memory/harness/codex.py`, initialize counters before reading:

```python
        input_tokens: int | None = None
        output_tokens: int | None = None
```

Inside the JSONL loop, before the `item.completed` filter continues, add:

```python
                if record.get("type") == "turn.completed":
                    usage = record.get("usage")
                    if isinstance(usage, dict):
                        raw_input = usage.get("input_tokens")
                        raw_output = usage.get("output_tokens")
                        if isinstance(raw_input, int):
                            input_tokens = raw_input
                        if isinstance(raw_output, int):
                            output_tokens = raw_output
                    continue
```

Return usage in the trajectory:

```python
        return Trajectory(
            user_prompt="",
            tool_calls=tool_calls,
            assistant_turns=assistant_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
```

- [ ] **Step 4: Capture Claude usage when available**

In `src/muscle_memory/harness/claude_code.py`, initialize counters before reading:

```python
        input_tokens: int | None = None
        output_tokens: int | None = None
```

Inside the JSONL loop after `rec` is parsed, add:

```python
                usage = rec.get("usage") or msg.get("usage")
                if isinstance(usage, dict):
                    raw_input = usage.get("input_tokens")
                    raw_output = usage.get("output_tokens")
                    if isinstance(raw_input, int):
                        input_tokens = raw_input
                    if isinstance(raw_output, int):
                        output_tokens = raw_output
```

Return usage in the trajectory:

```python
        return Trajectory(
            user_prompt=user_prompt,
            user_followup=" ".join(user_followups),
            tool_calls=tool_calls,
            assistant_turns=assistant_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
```

- [ ] **Step 5: Create task and measurement during transcript ingest**

In `src/muscle_memory/ingest.py`, import:

```python
from muscle_memory.personal_loop import add_measurement_for_task, capture_task
```

In `ingest_transcript_file()`, after `store.add_episode(episode)`, add:

```python
    task = capture_task(
        store,
        raw_prompt=episode.user_prompt,
        cleaned_prompt=episode.user_prompt,
        harness=transcript_format.replace("-jsonl", ""),
        project_path=episode.project_path,
        session_id=episode.session_id,
    )
    add_measurement_for_task(
        store,
        task=task,
        outcome=episode.outcome,
        reason="transcript ingest",
        input_tokens=episode.trajectory.input_tokens,
        output_tokens=episode.trajectory.output_tokens,
        injected_skill_tokens=0,
        tool_call_count=episode.trajectory.num_tool_calls(),
        comparable=episode.trajectory.num_tool_calls() > 0,
    )
```

In `ingest_episode_file()`, after `store.add_episode(episode)`, add the same shape with `harness=config.harness`.

- [ ] **Step 6: Run ingest tests and commit**

Run:

```bash
uv run pytest tests/test_ingest_measurement.py tests/test_ingest_codex.py tests/test_cli_harness.py::test_learn_transcript_records_episode_without_extraction -q
```

Expected: PASS.

Commit:

```bash
git add src/muscle_memory/harness/codex.py src/muscle_memory/harness/claude_code.py src/muscle_memory/ingest.py tests/test_ingest_measurement.py
git commit -m "feat: capture transcript measurement evidence"
```

---

### Task 5: Proof-Oriented `mm status`

**Files:**

- Modify: `src/muscle_memory/cli.py`
- Modify: `src/muscle_memory/personal_loop.py`
- Create: `tests/test_status_proof.py`
- Modify: `tests/test_cli_stats.py`

- [ ] **Step 1: Write failing status proof tests**

Create `tests/test_status_proof.py`:

```python
"""Tests for proof-oriented status metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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

runner = CliRunner()


def _config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
    )


def _task(store: Store, prompt: str, *, assisted: bool, outcome: Outcome, tokens: int) -> None:
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
            output_tokens=100,
            injected_skill_tokens=25 if assisted else 0,
            tool_call_count=2,
            comparable=True,
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
    assert proof["outcome_lift"] == 1.0
    assert proof["token_reduction"] == 0.3


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
```

- [ ] **Step 2: Run status proof tests and verify they fail**

Run:

```bash
uv run pytest tests/test_status_proof.py -q
```

Expected: FAIL because JSON has no `proof` key and rich output has no proof section.

- [ ] **Step 3: Add JSON proof payload helper**

In `src/muscle_memory/cli.py`, import `compute_proof_metrics` inside `stats()`:

```python
    from muscle_memory.personal_loop import compute_proof_metrics
```

After `retrieval_telemetry = ...`, compute:

```python
    proof = compute_proof_metrics(store)
```

In JSON output, add:

```python
            "proof": {
                "confidence": proof.confidence.value,
                "comparable_tasks": proof.comparable_tasks,
                "assisted_tasks": proof.assisted_tasks,
                "unassisted_tasks": proof.unassisted_tasks,
                "assisted_success_rate": proof.assisted_success_rate,
                "unassisted_success_rate": proof.unassisted_success_rate,
                "outcome_lift": proof.outcome_lift,
                "token_reduction": proof.token_reduction,
                "token_samples": proof.token_samples,
                "unknown_outcomes": proof.unknown_outcomes,
            },
```

- [ ] **Step 4: Render proof section in rich status**

In `stats()` after the header panel and before the empty store shortcut, add:

```python
    console.print(Rule("Proof"))
    if proof.comparable_tasks < 10:
        console.print(
            "  [yellow]insufficient evidence[/yellow]"
            f"  ({proof.comparable_tasks} comparable tasks)"
        )
        console.print("  [dim]Need at least 10 comparable measured tasks with assisted and unassisted examples.[/dim]")
    else:
        lift = (
            f"{proof.outcome_lift:+.1%}"
            if proof.outcome_lift is not None
            else "not enough paired outcomes"
        )
        token_delta = (
            f"{proof.token_reduction:.1%}"
            if proof.token_reduction is not None
            else "not enough token samples"
        )
        console.print(f"  [bold]outcome lift[/bold]    {lift}")
        console.print(f"  [bold]token reduction[/bold] {token_delta}")
        console.print(
            f"  [bold]confidence[/bold]      {proof.confidence.value}"
            f"  ({proof.comparable_tasks} comparable tasks)"
        )
        if proof.unknown_outcomes:
            console.print(f"  [yellow]unknown outcomes[/yellow] {proof.unknown_outcomes}")
```

Keep the existing empty-store shortcut after this section so a fresh user sees that proof is unavailable and then sees the onboarding prompt.

- [ ] **Step 5: Run status tests and commit**

Run:

```bash
uv run pytest tests/test_status_proof.py tests/test_cli_stats.py -q
```

Expected: PASS.

Commit:

```bash
git add src/muscle_memory/cli.py src/muscle_memory/personal_loop.py tests/test_status_proof.py tests/test_cli_stats.py
git commit -m "feat: show measurement proof in status"
```

---

### Task 6: Focused Docs And Full Verification

**Files:**

- Modify: `README.md`
- Modify: `docs/testing.md`

- [ ] **Step 1: Update README command list**

In `README.md`, change the quickstart section so it documents `mm use` instead of `mm retrieve` as the primary manual/Codex action:

```markdown
# use practiced skill for any harness, especially Codex
mm use "run the tests in this repo"

# compatibility search output for scripts
mm retrieve "run the tests in this repo" --json
```

In the primary loop list, use:

```markdown
- `mm use` emits compact practiced context for a task and records the activation.
- `mm status` shows whether the store is producing reuse, successful outcomes, and token savings with confidence labels.
```

- [ ] **Step 2: Update testing docs**

In `docs/testing.md`, add a measurement credibility line to the CLI test layer:

```markdown
- `tests/test_status_proof.py` verifies that `mm status` distinguishes insufficient evidence from confidence-labeled outcome/token proof.
- `tests/test_cli_use.py` verifies the Codex-friendly `mm use` path and canonical activation recording.
```

- [ ] **Step 3: Run targeted milestone verification**

Run:

```bash
uv run pytest tests/test_personal_loop_db.py tests/test_cli_use.py tests/test_hooks_personal_loop.py tests/test_ingest_measurement.py tests/test_status_proof.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run pytest -q
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/muscle_memory/
```

Expected:

- `pytest`: PASS, with the existing skipped tests unchanged.
- `ruff check`: no errors.
- `ruff format --check`: no formatting changes needed.
- `mypy`: no type errors.

- [ ] **Step 5: Commit docs and verification updates**

Commit:

```bash
git add README.md docs/testing.md
git commit -m "docs: document measurement credibility loop"
```

---

## Self-Review Checklist

Spec coverage:

- Task capture is implemented by Task 1 and Task 2.
- Skill activation delivery is implemented by Task 2 and Task 3.
- Outcome and token measurement is implemented by Task 3 and Task 4.
- Proof-oriented status is implemented by Task 5.
- Documentation drift is handled by Task 6.
- Team sharing, cloud sync, and Codex automatic hooks remain out of scope.

Type consistency:

- Model names are `TaskRecord`, `ActivationRecord`, and `MeasurementRecord`.
- Table names are `tasks`, `activations`, and `measurements`.
- Delivery modes are `claude-hook`, `codex-use`, and `manual`.
- Confidence values are `low`, `medium`, and `high`.

Final verification before handoff:

```bash
uv run pytest -q
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/muscle_memory/
```
