"""SQLite storage for Skills and Episodes.

This module is the only place that knows about SQL. Everything
else in muscle-memory talks to the `Store` DAO.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from muscle_memory.eval import EvalLabel

SCHEMA_VERSION = 8


def _l2_distance(a: list[float], b: list[float]) -> float:
    """Euclidean (L2) distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))


class Store:
    """SQLite-backed persistence for Skills and Episodes.

    All methods open and close a connection per call. For high-frequency
    operations, use `batch()` which keeps one connection open.
    """

    def __init__(self, db_path: Path, *, embedding_dims: int = 384):
        self.db_path = Path(db_path)
        self.embedding_dims = embedding_dims
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # connection management
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def batch(self) -> Iterator[sqlite3.Connection]:
        """Context manager that holds a single connection across multiple ops."""
        conn = self._open()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = self._open()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    activation TEXT NOT NULL,
                    execution TEXT NOT NULL,
                    termination TEXT NOT NULL,
                    tool_hints TEXT NOT NULL DEFAULT '[]',
                    tags TEXT NOT NULL DEFAULT '[]',
                    scope TEXT NOT NULL DEFAULT 'project',
                    score REAL NOT NULL DEFAULT 0.0,
                    invocations INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    maturity TEXT NOT NULL DEFAULT 'candidate',
                    source_episode_ids TEXT NOT NULL DEFAULT '[]',
                    refinement_count INTEGER NOT NULL DEFAULT 0,
                    previous_text TEXT,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    last_refined_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_skills_score ON skills(score DESC);
                CREATE INDEX IF NOT EXISTS idx_skills_maturity ON skills(maturity);
                CREATE INDEX IF NOT EXISTS idx_skills_scope ON skills(scope);

                CREATE TABLE IF NOT EXISTS episodes (
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

                CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
                CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at DESC);

                CREATE TABLE IF NOT EXISTS eval_labels (
                    id TEXT PRIMARY KEY,
                    label_type TEXT NOT NULL,
                    episode_id TEXT,
                    heuristic_outcome TEXT,
                    human_outcome TEXT,
                    confidence TEXT DEFAULT 'high',
                    notes TEXT DEFAULT '',
                    query_text TEXT,
                    skill_id TEXT,
                    relevance TEXT,
                    rank_position INTEGER,
                    labeled_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_labels_unique
                    ON eval_labels(label_type, episode_id, skill_id);

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind);
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_single_active_refine
                    ON jobs(kind)
                    WHERE kind = 'refine' AND status IN ('pending', 'running');

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
                CREATE INDEX IF NOT EXISTS idx_tasks_session_created
                    ON tasks(session_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS activations (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    skill_id TEXT NOT NULL,
                    distance REAL,
                    final_rank REAL,
                    delivery_mode TEXT NOT NULL,
                    injected_token_count INTEGER NOT NULL DEFAULT 0
                        CHECK (injected_token_count >= 0),
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
                    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
                    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
                    injected_skill_tokens INTEGER NOT NULL DEFAULT 0
                        CHECK (injected_skill_tokens >= 0),
                    tool_call_count INTEGER NOT NULL DEFAULT 0
                        CHECK (tool_call_count >= 0),
                    comparable INTEGER NOT NULL DEFAULT 0,
                    measured_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_measurements_outcome ON measurements(outcome);
                CREATE INDEX IF NOT EXISTS idx_measurements_comparable ON measurements(comparable);
                CREATE INDEX IF NOT EXISTS idx_measurements_measured_at
                    ON measurements(measured_at DESC);
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_embeddings (
                    skill_id TEXT PRIMARY KEY REFERENCES skills(id) ON DELETE CASCADE,
                    embedding TEXT NOT NULL
                )
                """
            )

            cur = conn.execute("SELECT version FROM schema_version")
            row = cur.fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            else:
                self._migrate(conn, current_version=int(row["version"]))
            conn.commit()
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection, *, current_version: int) -> None:
        """Apply idempotent ALTER TABLE migrations for schema version bumps.

        SQLite supports `ADD COLUMN` but not `ADD COLUMN IF NOT EXISTS`,
        so we check existing columns before altering.
        """
        if current_version >= SCHEMA_VERSION:
            return

        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(skills)").fetchall()
        }

        # v2 — refinement state columns on skills
        if "refinement_count" not in existing_cols:
            conn.execute(
                "ALTER TABLE skills ADD COLUMN refinement_count INTEGER NOT NULL DEFAULT 0"
            )
        if "previous_text" not in existing_cols:
            conn.execute("ALTER TABLE skills ADD COLUMN previous_text TEXT")

        # v3 — replace sqlite-vec virtual table with plain embeddings table
        if current_version < 3:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_embeddings (
                    skill_id TEXT PRIMARY KEY REFERENCES skills(id) ON DELETE CASCADE,
                    embedding TEXT NOT NULL
                )
                """
            )
            # Try to drop the old sqlite-vec virtual table. This may fail
            # if the vec0 module isn't loaded (the whole reason we're
            # migrating away from it), so we just ignore the error.
            try:
                conn.execute("DROP TABLE IF EXISTS skill_vec")
            except Exception:
                pass  # vec0 module not available, table is orphaned but harmless

        # v4 — eval labels table
        if current_version < 4:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_labels (
                    id TEXT PRIMARY KEY,
                    label_type TEXT NOT NULL,
                    episode_id TEXT,
                    heuristic_outcome TEXT,
                    human_outcome TEXT,
                    confidence TEXT DEFAULT 'high',
                    notes TEXT DEFAULT '',
                    query_text TEXT,
                    skill_id TEXT,
                    relevance TEXT,
                    rank_position INTEGER,
                    labeled_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_labels_unique "
                "ON eval_labels(label_type, episode_id)"
            )

        # v5 — tracked async jobs
        if current_version < 5:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")

        if current_version < 6:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_single_active_refine
                    ON jobs(kind)
                    WHERE kind = 'refine' AND status IN ('pending', 'running')
                """
            )

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
                CREATE INDEX IF NOT EXISTS idx_tasks_session_created
                    ON tasks(session_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS activations (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    skill_id TEXT NOT NULL,
                    distance REAL,
                    final_rank REAL,
                    delivery_mode TEXT NOT NULL,
                    injected_token_count INTEGER NOT NULL DEFAULT 0
                        CHECK (injected_token_count >= 0),
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
                    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
                    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
                    injected_skill_tokens INTEGER NOT NULL DEFAULT 0
                        CHECK (injected_skill_tokens >= 0),
                    tool_call_count INTEGER NOT NULL DEFAULT 0
                        CHECK (tool_call_count >= 0),
                    comparable INTEGER NOT NULL DEFAULT 0,
                    measured_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_measurements_outcome ON measurements(outcome);
                CREATE INDEX IF NOT EXISTS idx_measurements_comparable ON measurements(comparable);
                CREATE INDEX IF NOT EXISTS idx_measurements_measured_at
                    ON measurements(measured_at DESC);
                """
            )

        if current_version == 7:
            self._validate_v7_personal_loop_constraints(conn)
            self._rebuild_activations_with_constraints(conn)
            self._rebuild_measurements_with_constraints(conn)

        if current_version < 8:
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_session_created
                    ON tasks(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_measurements_measured_at
                    ON measurements(measured_at DESC);
                """
            )

        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def _validate_v7_personal_loop_constraints(self, conn: sqlite3.Connection) -> None:
        activation = conn.execute(
            """
            SELECT id FROM activations
            WHERE injected_token_count < 0
            LIMIT 1
            """
        ).fetchone()
        if activation is not None:
            raise RuntimeError(
                "Cannot migrate v7 personal loop tables: "
                "activations.injected_token_count contains negative values"
            )

        measurement = conn.execute(
            """
            SELECT id FROM measurements
            WHERE input_tokens < 0
               OR output_tokens < 0
               OR injected_skill_tokens < 0
               OR tool_call_count < 0
            LIMIT 1
            """
        ).fetchone()
        if measurement is not None:
            raise RuntimeError(
                "Cannot migrate v7 personal loop tables: "
                "measurements token/count columns contain negative values"
            )

    def _rebuild_activations_with_constraints(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            ALTER TABLE activations RENAME TO activations_old;

            CREATE TABLE activations (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                skill_id TEXT NOT NULL,
                distance REAL,
                final_rank REAL,
                delivery_mode TEXT NOT NULL,
                injected_token_count INTEGER NOT NULL DEFAULT 0
                    CHECK (injected_token_count >= 0),
                credited_outcome TEXT,
                created_at TEXT NOT NULL,
                credited_at TEXT
            );

            INSERT INTO activations (
                id, task_id, skill_id, distance, final_rank, delivery_mode,
                injected_token_count, credited_outcome, created_at, credited_at
            )
            SELECT
                id, task_id, skill_id, distance, final_rank, delivery_mode,
                injected_token_count, credited_outcome, created_at, credited_at
            FROM activations_old;

            DROP TABLE activations_old;

            CREATE INDEX IF NOT EXISTS idx_activations_task ON activations(task_id);
            CREATE INDEX IF NOT EXISTS idx_activations_skill ON activations(skill_id);
            """
        )

    def _rebuild_measurements_with_constraints(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            ALTER TABLE measurements RENAME TO measurements_old;

            CREATE TABLE measurements (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE REFERENCES tasks(id) ON DELETE CASCADE,
                outcome TEXT NOT NULL DEFAULT 'unknown',
                confidence TEXT NOT NULL DEFAULT 'low',
                reason TEXT NOT NULL DEFAULT '',
                input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
                output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
                injected_skill_tokens INTEGER NOT NULL DEFAULT 0
                    CHECK (injected_skill_tokens >= 0),
                tool_call_count INTEGER NOT NULL DEFAULT 0
                    CHECK (tool_call_count >= 0),
                comparable INTEGER NOT NULL DEFAULT 0,
                measured_at TEXT NOT NULL
            );

            INSERT INTO measurements (
                id, task_id, outcome, confidence, reason, input_tokens,
                output_tokens, injected_skill_tokens, tool_call_count,
                comparable, measured_at
            )
            SELECT
                id, task_id, outcome, confidence, reason, input_tokens,
                output_tokens, injected_skill_tokens, tool_call_count,
                comparable, measured_at
            FROM measurements_old;

            DROP TABLE measurements_old;

            CREATE INDEX IF NOT EXISTS idx_measurements_outcome ON measurements(outcome);
            CREATE INDEX IF NOT EXISTS idx_measurements_comparable ON measurements(comparable);
            CREATE INDEX IF NOT EXISTS idx_measurements_measured_at
                ON measurements(measured_at DESC);
            """
        )

    # ------------------------------------------------------------------
    # skills
    # ------------------------------------------------------------------

    def add_skill(self, skill: Skill, *, embedding: list[float] | None = None) -> None:
        with self.batch() as conn:
            self._insert_skill(conn, skill, embedding)

    def _insert_skill(
        self,
        conn: sqlite3.Connection,
        skill: Skill,
        embedding: list[float] | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO skills (
                id, activation, execution, termination, tool_hints, tags, scope,
                score, invocations, successes, failures, maturity,
                source_episode_ids, refinement_count, previous_text,
                created_at, last_used_at, last_refined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill.id,
                skill.activation,
                skill.execution,
                skill.termination,
                _dumps(skill.tool_hints),
                _dumps(skill.tags),
                skill.scope.value,
                skill.score,
                skill.invocations,
                skill.successes,
                skill.failures,
                skill.maturity.value,
                _dumps(skill.source_episode_ids),
                skill.refinement_count,
                _dumps(skill.previous_text) if skill.previous_text else None,
                _iso(skill.created_at),
                _iso(skill.last_used_at),
                _iso(skill.last_refined_at),
            ),
        )
        if embedding is not None:
            self._ensure_embedding_dim(embedding)
            conn.execute(
                "INSERT INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)",
                (skill.id, json.dumps(embedding)),
            )

    def _ensure_embedding_dim(self, embedding: list[float]) -> None:
        if len(embedding) != self.embedding_dims:
            raise ValueError(
                f"Embedding dim mismatch: got {len(embedding)}, "
                f"expected {self.embedding_dims}. "
                "Reinitialize the store with the correct embedding_dims."
            )

    def update_skill(self, skill: Skill, *, embedding: list[float] | None = None) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                UPDATE skills SET
                    activation = ?,
                    execution = ?,
                    termination = ?,
                    tool_hints = ?,
                    tags = ?,
                    scope = ?,
                    score = ?,
                    invocations = ?,
                    successes = ?,
                    failures = ?,
                    maturity = ?,
                    source_episode_ids = ?,
                    refinement_count = ?,
                    previous_text = ?,
                    last_used_at = ?,
                    last_refined_at = ?
                WHERE id = ?
                """,
                (
                    skill.activation,
                    skill.execution,
                    skill.termination,
                    _dumps(skill.tool_hints),
                    _dumps(skill.tags),
                    skill.scope.value,
                    skill.score,
                    skill.invocations,
                    skill.successes,
                    skill.failures,
                    skill.maturity.value,
                    _dumps(skill.source_episode_ids),
                    skill.refinement_count,
                    _dumps(skill.previous_text) if skill.previous_text else None,
                    _iso(skill.last_used_at),
                    _iso(skill.last_refined_at),
                    skill.id,
                ),
            )
            if embedding is not None:
                self._ensure_embedding_dim(embedding)
                conn.execute(
                    "INSERT OR REPLACE INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)",
                    (skill.id, json.dumps(embedding)),
                )

    def get_skill(self, skill_id: str) -> Skill | None:
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
            return _row_to_skill(row) if row else None
        finally:
            conn.close()

    def delete_skill(self, skill_id: str) -> None:
        with self.batch() as conn:
            conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            conn.execute("DELETE FROM skill_embeddings WHERE skill_id = ?", (skill_id,))

    def list_skills(
        self,
        *,
        scope: Scope | None = None,
        maturity: Maturity | None = None,
        limit: int | None = None,
    ) -> list[Skill]:
        sql = "SELECT * FROM skills WHERE 1=1"
        params: list[Any] = []
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope.value)
        if maturity is not None:
            if maturity is Maturity.LIVE:
                sql += " AND maturity IN (?, ?)"
                params.extend([Maturity.LIVE.value, "established"])
            else:
                sql += " AND maturity = ?"
                params.append(maturity.value)
        sql += " ORDER BY score DESC, invocations DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        conn = self._open()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_skill(r) for r in rows]
        finally:
            conn.close()

    def count_skills(self, *, scope: Scope | None = None) -> int:
        conn = self._open()
        try:
            if scope is not None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM skills WHERE scope = ?", (scope.value,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()
            return int(row["n"])
        finally:
            conn.close()

    def count_skills_since(self, iso_timestamp: str) -> int:
        """Count skills created after the given ISO-8601 timestamp."""
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM skills WHERE created_at > ?",
                (iso_timestamp,),
            ).fetchone()
            return int(row["n"])
        finally:
            conn.close()

    def search_skills_by_embedding(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        scope: Scope | None = None,
    ) -> list[tuple[Skill, float]]:
        """KNN search over activation embeddings.

        Returns a list of (Skill, distance) tuples ordered by ascending
        distance (closer = more similar). Uses brute-force L2 distance.
        """
        self._ensure_embedding_dim(embedding)

        conn = self._open()
        try:
            rows = conn.execute("SELECT skill_id, embedding FROM skill_embeddings").fetchall()

            if not rows:
                return []

            # Brute-force L2 distance (skip corrupted embeddings)
            distances: list[tuple[str, float]] = []
            for row in rows:
                try:
                    stored = json.loads(row["embedding"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if len(stored) != self.embedding_dims:
                    continue
                distances.append((row["skill_id"], _l2_distance(embedding, stored)))

            distances.sort(key=lambda t: t[1])
            candidates = distances[: top_k * 4]

            # Load full Skills for the hits, filter by scope if requested
            by_id = {sid: dist for sid, dist in candidates}
            placeholders = ",".join("?" * len(by_id))
            sql = f"SELECT * FROM skills WHERE id IN ({placeholders})"
            params: list[Any] = list(by_id.keys())
            if scope is not None:
                sql += " AND scope = ?"
                params.append(scope.value)

            skill_rows = conn.execute(sql, params).fetchall()
            results = [(_row_to_skill(r), by_id[r["id"]]) for r in skill_rows]
            results.sort(key=lambda t: t[1])
            return results[:top_k]
        finally:
            conn.close()

    def get_skill_embedding(self, skill_id: str) -> list[float] | None:
        """Load the stored embedding vector for a single skill."""
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT embedding FROM skill_embeddings WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
            if row is None:
                return None
            loaded = json.loads(row["embedding"])
            if not isinstance(loaded, list):
                return None
            return [float(value) for value in loaded]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # episodes
    # ------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    id, session_id, user_prompt, trajectory, outcome, reward,
                    started_at, ended_at, project_path, activated_skills
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.session_id,
                    episode.user_prompt,
                    episode.trajectory.model_dump_json(),
                    episode.outcome.value,
                    episode.reward,
                    _iso(episode.started_at),
                    _iso(episode.ended_at),
                    episode.project_path,
                    _dumps(episode.activated_skills),
                ),
            )

    def update_episode_outcome(self, episode_id: str, *, outcome: Outcome, reward: float) -> None:
        """Update just the outcome/reward of an existing episode.

        Used by `mm rescore` after the outcome heuristic changes.
        """
        with self.batch() as conn:
            conn.execute(
                "UPDATE episodes SET outcome = ?, reward = ? WHERE id = ?",
                (outcome.value, reward, episode_id),
            )

    def get_episode(self, episode_id: str) -> Episode | None:
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            return _row_to_episode(row) if row else None
        finally:
            conn.close()

    def list_episodes(self, *, limit: int | None = 50) -> list[Episode]:
        conn = self._open()
        try:
            if limit is None:
                rows = conn.execute("SELECT * FROM episodes ORDER BY started_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [_row_to_episode(r) for r in rows]
        finally:
            conn.close()

    def count_episodes(self) -> int:
        """Return the total number of episodes in the store."""
        conn = self._open()
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
            return int(row["n"])
        finally:
            conn.close()

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
                """
                SELECT * FROM tasks
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
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
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [_row_to_task(r) for r in rows]
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
            return [_row_to_activation(r) for r in rows]
        finally:
            conn.close()

    def list_task_ids_with_activations(self) -> set[str]:
        conn = self._open()
        try:
            rows = conn.execute("SELECT DISTINCT task_id FROM activations").fetchall()
            return {str(row["task_id"]) for row in rows}
        finally:
            conn.close()

    def credit_activations(self, task_id: str, skill_ids: list[str], outcome: Outcome) -> None:
        if not skill_ids:
            return
        placeholders = ",".join("?" for _ in skill_ids)
        with self.batch() as conn:
            conn.execute(
                f"""
                UPDATE activations
                SET credited_outcome = ?, credited_at = ?
                WHERE task_id = ? AND skill_id IN ({placeholders})
                """,
                (outcome.value, _iso(datetime.now(UTC)), task_id, *skill_ids),
            )

    def add_measurement(self, measurement: MeasurementRecord) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                INSERT INTO measurements (
                    id, task_id, outcome, confidence, reason, input_tokens,
                    output_tokens, injected_skill_tokens, tool_call_count,
                    comparable, measured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    outcome = excluded.outcome,
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    injected_skill_tokens = excluded.injected_skill_tokens,
                    tool_call_count = excluded.tool_call_count,
                    comparable = excluded.comparable,
                    measured_at = excluded.measured_at
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
                    int(measurement.comparable),
                    _iso(measurement.measured_at),
                ),
            )

    def get_measurement_for_task(self, task_id: str) -> MeasurementRecord | None:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM measurements WHERE task_id = ?", (task_id,)
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
            return [_row_to_measurement(r) for r in rows]
        finally:
            conn.close()

    def find_episodes_for_skill(self, skill_id: str, *, limit: int = 10) -> list[Episode]:
        """Return the most recent episodes where this skill was activated.

        Used by the refinement loop to look up trajectories for
        semantic gradient extraction and PPO-Gate verification.
        Walks recent episodes in memory (activated_skills is a JSON
        list, not indexed) — fine for reasonable episode counts.
        """
        conn = self._open()
        try:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
                (limit * 10,),  # scan more to find enough matches
            ).fetchall()
            matches: list[Episode] = []
            for row in rows:
                activated = json.loads(row["activated_skills"] or "[]")
                if skill_id in activated:
                    matches.append(_row_to_episode(row))
                    if len(matches) >= limit:
                        break
            return matches
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # eval labels
    # ------------------------------------------------------------------

    def add_eval_label(self, label: EvalLabel) -> None:
        conn = self._open()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO eval_labels
                   (id, label_type, episode_id, heuristic_outcome, human_outcome,
                    confidence, notes, query_text, skill_id, relevance,
                    rank_position, labeled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    label.id,
                    label.label_type,
                    label.episode_id,
                    label.heuristic_outcome,
                    label.human_outcome,
                    label.confidence,
                    label.notes,
                    label.query_text,
                    label.skill_id,
                    label.relevance,
                    label.rank_position,
                    label.labeled_at.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_eval_labels(self, label_type: str, *, limit: int = 500) -> list[EvalLabel]:
        conn = self._open()
        try:
            rows = conn.execute(
                "SELECT * FROM eval_labels WHERE label_type = ? ORDER BY labeled_at DESC LIMIT ?",
                (label_type, limit),
            ).fetchall()
            return [_row_to_eval_label(r) for r in rows]
        finally:
            conn.close()

    def get_outcome_label(self, episode_id: str) -> EvalLabel | None:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM eval_labels WHERE label_type = 'outcome' AND episode_id = ?",
                (episode_id,),
            ).fetchone()
            return _row_to_eval_label(row) if row else None
        finally:
            conn.close()

    def count_eval_labels(self, label_type: str) -> int:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM eval_labels WHERE label_type = ?",
                (label_type,),
            ).fetchone()
            return int(row["n"])
        finally:
            conn.close()

    def list_unlabeled_episodes(self, *, limit: int = 50) -> list[Episode]:
        conn = self._open()
        try:
            rows = conn.execute(
                """SELECT e.* FROM episodes e
                   LEFT JOIN eval_labels el
                     ON el.label_type = 'outcome' AND el.episode_id = e.id
                   WHERE el.id IS NULL
                   ORDER BY e.started_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [_row_to_episode(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------

    def add_job(self, job: BackgroundJob) -> None:
        with self.batch() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, kind, status, payload, attempts, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.kind.value,
                    job.status.value,
                    _dumps(job.payload),
                    job.attempts,
                    job.error,
                    _iso(job.created_at),
                    _iso(job.updated_at),
                ),
            )

    def get_job(self, job_id: str) -> BackgroundJob | None:
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return _row_to_job(row) if row else None
        finally:
            conn.close()

    def list_jobs(
        self,
        *,
        limit: int | None = 50,
        status: JobStatus | None = None,
        kind: JobKind | None = None,
    ) -> list[BackgroundJob]:
        conn = self._open()
        try:
            sql = "SELECT * FROM jobs"
            params: list[Any] = []
            clauses: list[str] = []
            if status is not None:
                clauses.append("status = ?")
                params.append(status.value)
            if kind is not None:
                clauses.append("kind = ?")
                params.append(kind.value)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY created_at DESC"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [_row_to_job(r) for r in rows]
        finally:
            conn.close()

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by id. Returns True if a row was removed."""
        with self.batch() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def delete_jobs_by_status(self, status: JobStatus) -> int:
        """Delete every job with the given status. Returns rows removed."""
        with self.batch() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE status = ?", (status.value,))
            return cur.rowcount

    def update_job_status(
        self,
        job_id: str,
        *,
        status: JobStatus,
        attempts: int | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.batch() as conn:
            existing = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if existing is None:
                raise KeyError(f"job not found: {job_id}")
            next_attempts = int(existing["attempts"]) if attempts is None else attempts
            next_error = error
            if error is None and status is not JobStatus.FAILED:
                next_error = None
            elif error is None:
                next_error = existing["error"]
            next_payload = existing["payload"] if payload is None else _dumps(payload)
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, attempts = ?, error = ?, payload = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    next_attempts,
                    next_error,
                    next_payload,
                    _iso(datetime.now(UTC)),
                    job_id,
                ),
            )


# ----------------------------------------------------------------------
# row -> model hydration helpers
# ----------------------------------------------------------------------


def _row_to_job(row: sqlite3.Row) -> BackgroundJob:
    return BackgroundJob(
        id=row["id"],
        kind=JobKind(row["kind"]),
        status=JobStatus(row["status"]),
        payload=json.loads(row["payload"] or "{}"),
        attempts=int(row["attempts"]),
        error=row["error"],
        created_at=_parse_iso(row["created_at"]) or datetime.now(UTC),
        updated_at=_parse_iso(row["updated_at"]) or datetime.now(UTC),
    )


def _row_to_skill(row: sqlite3.Row) -> Skill:
    # `previous_text` and `refinement_count` were added in schema v2;
    # sqlite3.Row access for a missing column returns IndexError, so
    # guard with a fallback to support older DBs mid-migration.
    try:
        refinement_count = row["refinement_count"]
    except (IndexError, KeyError):
        refinement_count = 0
    try:
        prev_raw = row["previous_text"]
    except (IndexError, KeyError):
        prev_raw = None
    previous_text = json.loads(prev_raw) if prev_raw else None

    return Skill(
        id=row["id"],
        activation=row["activation"],
        execution=row["execution"],
        termination=row["termination"],
        tool_hints=json.loads(row["tool_hints"]),
        tags=json.loads(row["tags"]),
        scope=Scope(row["scope"]),
        score=row["score"],
        invocations=row["invocations"],
        successes=row["successes"],
        failures=row["failures"],
        maturity=Maturity(row["maturity"]),
        source_episode_ids=json.loads(row["source_episode_ids"]),
        refinement_count=refinement_count,
        previous_text=previous_text,
        created_at=_parse_iso(row["created_at"]) or datetime.now(UTC),
        last_used_at=_parse_iso(row["last_used_at"]),
        last_refined_at=_parse_iso(row["last_refined_at"]),
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    traj_data = json.loads(row["trajectory"])
    trajectory = Trajectory(
        user_prompt=traj_data.get("user_prompt", ""),
        user_followup=traj_data.get("user_followup", ""),
        tool_calls=[ToolCall(**tc) for tc in traj_data.get("tool_calls", [])],
        assistant_turns=traj_data.get("assistant_turns", []),
        input_tokens=traj_data.get("input_tokens"),
        output_tokens=traj_data.get("output_tokens"),
    )
    return Episode(
        id=row["id"],
        session_id=row["session_id"],
        user_prompt=row["user_prompt"],
        trajectory=trajectory,
        outcome=Outcome(row["outcome"]),
        reward=row["reward"],
        started_at=_parse_iso(row["started_at"]) or datetime.now(UTC),
        ended_at=_parse_iso(row["ended_at"]),
        project_path=row["project_path"],
        activated_skills=json.loads(row["activated_skills"]),
    )


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
    credited_outcome = row["credited_outcome"]
    return ActivationRecord(
        id=row["id"],
        task_id=row["task_id"],
        skill_id=row["skill_id"],
        distance=row["distance"],
        final_rank=row["final_rank"],
        delivery_mode=DeliveryMode(row["delivery_mode"]),
        injected_token_count=int(row["injected_token_count"]),
        credited_outcome=Outcome(credited_outcome) if credited_outcome else None,
        created_at=_parse_iso(row["created_at"]) or datetime.now(UTC),
        credited_at=_parse_iso(row["credited_at"]),
    )


def _row_to_measurement(row: sqlite3.Row) -> MeasurementRecord:
    return MeasurementRecord(
        id=row["id"],
        task_id=row["task_id"],
        outcome=Outcome(row["outcome"]),
        confidence=EvidenceConfidence(row["confidence"]),
        reason=row["reason"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        injected_skill_tokens=int(row["injected_skill_tokens"]),
        tool_call_count=int(row["tool_call_count"]),
        comparable=bool(row["comparable"]),
        measured_at=_parse_iso(row["measured_at"]) or datetime.now(UTC),
    )


def _row_to_eval_label(row: sqlite3.Row) -> EvalLabel:
    from muscle_memory.eval import EvalLabel as _EvalLabel

    return _EvalLabel(
        id=row["id"],
        label_type=row["label_type"],
        episode_id=row["episode_id"] or "",
        heuristic_outcome=row["heuristic_outcome"] or "",
        human_outcome=row["human_outcome"] or "",
        confidence=row["confidence"] or "high",
        notes=row["notes"] or "",
        query_text=row["query_text"] or "",
        skill_id=row["skill_id"] or "",
        relevance=row["relevance"] or "",
        rank_position=row["rank_position"] or 0,
        labeled_at=_parse_iso(row["labeled_at"]) or datetime.now(UTC),
    )
