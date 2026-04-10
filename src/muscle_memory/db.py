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
from datetime import datetime
from pathlib import Path
from typing import Any

from muscle_memory.models import (
    Episode,
    Maturity,
    Outcome,
    Scope,
    Skill,
    ToolCall,
    Trajectory,
)

SCHEMA_VERSION = 4


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
                    ON eval_labels(label_type, episode_id);
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

        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

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

    def list_episodes(self, *, limit: int = 50) -> list[Episode]:
        conn = self._open()
        try:
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


# ----------------------------------------------------------------------
# row -> model hydration helpers
# ----------------------------------------------------------------------


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
        created_at=_parse_iso(row["created_at"]),  # type: ignore[arg-type]
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
    )
    return Episode(
        id=row["id"],
        session_id=row["session_id"],
        user_prompt=row["user_prompt"],
        trajectory=trajectory,
        outcome=Outcome(row["outcome"]),
        reward=row["reward"],
        started_at=_parse_iso(row["started_at"]),  # type: ignore[arg-type]
        ended_at=_parse_iso(row["ended_at"]),
        project_path=row["project_path"],
        activated_skills=json.loads(row["activated_skills"]),
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
        labeled_at=_parse_iso(row["labeled_at"]) or datetime.now(),  # type: ignore[arg-type]
    )
