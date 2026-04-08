"""Edge case tests covering tricky inputs, adversarial data, and
environment quirks that unit tests don't exercise.

Failure modes we care about:
- Malformed / hostile input never crashes the hook
- Unicode round-trips cleanly (no mojibake)
- Storage can hold the biggest skills we'd realistically see
- Concurrent writes don't corrupt the DB
- Schema drift is handled gracefully
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from muscle_memory.db import Store
from muscle_memory.hooks.stop import main as stop_main, parse_transcript
from muscle_memory.hooks.user_prompt import main as user_prompt_main
from muscle_memory.models import (
    Episode,
    Maturity,
    Outcome,
    Scope,
    Skill,
    ToolCall,
    Trajectory,
)
from muscle_memory.outcomes import infer_outcome


# ----------------------------------------------------------------------
# Malformed / hostile input never crashes hooks
# ----------------------------------------------------------------------


class TestHookResilience:
    def test_user_prompt_hook_survives_all_garbage_stdin(self) -> None:
        """Any bad input → rc 0, no output, no crash."""
        cases = [
            "",  # empty
            "not json",
            "{",  # incomplete
            "null",
            "[]",  # valid JSON but wrong shape
            '{"unrelated": "thing"}',  # valid JSON but no prompt
            '{"prompt": null}',  # null prompt
            '{"prompt": 12345}',  # wrong type
            '\x00\x01\x02',  # binary garbage
        ]
        for case in cases:
            stdin = StringIO(case)
            stdout = StringIO()
            with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
                rc = user_prompt_main()
            assert rc == 0, f"crashed on: {case!r}"
            # either empty output or at most a valid injection
            out = stdout.getvalue()
            if out:
                assert "<muscle_memory>" in out or out.strip() == ""

    def test_stop_hook_survives_all_garbage_stdin(self) -> None:
        cases = [
            "",
            "not json",
            '{"session_id": null}',
            '{"transcript_path": "/nowhere/at/all"}',
            '{"session_id": "x", "transcript_path": 123}',
            '\x00\x01\x02',
        ]
        for case in cases:
            stdin = StringIO(case)
            with patch("sys.stdin", stdin):
                rc = stop_main()
            assert rc == 0, f"crashed on: {case!r}"

    def test_stop_hook_nonexistent_transcript_is_silent(self) -> None:
        payload = {
            "session_id": "x",
            "cwd": "/tmp",
            "transcript_path": "/path/that/does/not/exist.jsonl",
        }
        stdin = StringIO(json.dumps(payload))
        with patch("sys.stdin", stdin):
            rc = stop_main()
        assert rc == 0

    def test_stop_hook_transcript_with_only_comments(self, tmp_path: Path) -> None:
        """JSONL files can have blank lines. Parse shouldn't care."""
        path = tmp_path / "t.jsonl"
        path.write_text("\n\n\n   \n\n")
        traj = parse_transcript(path)
        assert traj.tool_calls == []
        assert traj.user_prompt == ""


# ----------------------------------------------------------------------
# Unicode / emoji / international characters
# ----------------------------------------------------------------------


class TestUnicode:
    def test_skill_with_emoji_and_unicode_round_trips(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "mm.db", embedding_dims=4)
            skill = Skill(
                activation="When tests fail 🧪 and the user says 不能工作",
                execution="1. Läuft die Pipeline? 🚀\n2. Vérifie les logs\n3. ¡Reinicia!",
                termination="Tests pass ✅ and 日志 are clean",
                tags=["i18n", "émoji", "中文"],
            )
            store.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])
            loaded = store.get_skill(skill.id)
            assert loaded is not None
            assert loaded.activation == skill.activation
            assert loaded.execution == skill.execution
            assert loaded.termination == skill.termination
            assert loaded.tags == skill.tags

    def test_trajectory_with_unicode_tool_output(self) -> None:
        traj = Trajectory(
            tool_calls=[
                ToolCall(
                    name="Bash",
                    arguments={"command": "echo 'café'"},
                    result="café\n日本語\n🎉",
                )
            ]
        )
        sig = infer_outcome(traj)
        # just ensure no crash during keyword matching
        assert sig.outcome in (Outcome.SUCCESS, Outcome.FAILURE, Outcome.UNKNOWN)

    def test_transcript_with_utf8_content(self, tmp_path: Path) -> None:
        path = tmp_path / "utf8.jsonl"
        path.write_text(
            json.dumps(
                {"type": "user", "message": {"content": "Bonjour, comment ça va? 🇫🇷"}},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        traj = parse_transcript(path)
        assert "Bonjour" in traj.user_prompt
        assert "🇫🇷" in traj.user_prompt


# ----------------------------------------------------------------------
# SQL injection-like input / adversarial strings
# ----------------------------------------------------------------------


class TestAdversarialInput:
    def test_skill_with_sql_injection_attempt_in_activation(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "mm.db", embedding_dims=4)
            skill = Skill(
                activation="'; DROP TABLE skills; --",
                execution="normal steps",
                termination="done",
            )
            store.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])
            # the store should still exist and work
            loaded = store.get_skill(skill.id)
            assert loaded is not None
            assert loaded.activation == "'; DROP TABLE skills; --"
            assert store.count_skills() == 1

    def test_skill_with_null_bytes_in_text(self) -> None:
        """Pydantic validation should accept, SQLite should accept."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "mm.db", embedding_dims=4)
            skill = Skill(
                activation="hello\x00world",
                execution="step 1\x00step 2",
                termination="done\x00",
            )
            store.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])
            loaded = store.get_skill(skill.id)
            assert loaded is not None
            # SQLite round-trips null bytes in TEXT columns
            assert "\x00" in loaded.activation

    def test_skill_with_very_long_activation(self) -> None:
        """~10KB activation text should still be storable."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "mm.db", embedding_dims=4)
            long_text = "This is a very long activation condition. " * 250
            assert len(long_text) > 10_000
            skill = Skill(
                activation=long_text,
                execution="short execution",
                termination="done",
            )
            store.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])
            loaded = store.get_skill(skill.id)
            assert loaded is not None
            assert len(loaded.activation) > 10_000


# ----------------------------------------------------------------------
# Schema / validation rejection
# ----------------------------------------------------------------------


class TestValidation:
    def test_empty_activation_raises(self) -> None:
        with pytest.raises(ValueError):
            Skill(activation="", execution="x", termination="y")

    def test_whitespace_only_activation_raises(self) -> None:
        with pytest.raises(ValueError):
            Skill(activation="   \t\n  ", execution="x", termination="y")

    def test_skill_fields_are_trimmed(self) -> None:
        s = Skill(
            activation="   when tests fail   ",
            execution="  do x  ",
            termination="  done  ",
        )
        assert s.activation == "when tests fail"
        assert s.execution == "do x"
        assert s.termination == "done"


# ----------------------------------------------------------------------
# Embedding dim mismatch and related storage edge cases
# ----------------------------------------------------------------------


class TestStorageEdgeCases:
    def test_dim_mismatch_on_insert_raises(self, tmp_db: Store) -> None:
        skill = Skill(activation="a", execution="b", termination="c")
        with pytest.raises(ValueError, match="Embedding dim mismatch"):
            tmp_db.add_skill(skill, embedding=[0.1, 0.2])  # wrong dim

    def test_dim_mismatch_on_search_raises(self, tmp_db: Store) -> None:
        with pytest.raises(ValueError, match="Embedding dim mismatch"):
            tmp_db.search_skills_by_embedding([0.1, 0.2])  # wrong dim

    def test_search_empty_store_returns_empty(self, tmp_db: Store) -> None:
        hits = tmp_db.search_skills_by_embedding([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert hits == []

    def test_list_skills_empty_store(self, tmp_db: Store) -> None:
        assert tmp_db.list_skills() == []
        assert tmp_db.count_skills() == 0

    def test_delete_nonexistent_skill_is_noop(self, tmp_db: Store) -> None:
        tmp_db.delete_skill("does-not-exist-id")
        # no exception
        assert tmp_db.count_skills() == 0

    def test_get_nonexistent_skill_returns_none(self, tmp_db: Store) -> None:
        assert tmp_db.get_skill("nope") is None

    def test_get_nonexistent_episode_returns_none(self, tmp_db: Store) -> None:
        assert tmp_db.get_episode("nope") is None


# ----------------------------------------------------------------------
# Concurrency: two writers at once
# ----------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_skill_inserts_do_not_corrupt(self, tmp_path: Path) -> None:
        """Two threads inserting skills simultaneously should both succeed
        and the DB should have both skills at the end. SQLite in WAL mode
        handles this."""
        db_path = tmp_path / "concurrent.db"
        store = Store(db_path, embedding_dims=4)

        errors: list[Exception] = []

        def insert(n: int, offset: float) -> None:
            try:
                for i in range(n):
                    # each worker gets its own Store to avoid sharing connections
                    worker_store = Store(db_path, embedding_dims=4)
                    worker_store.add_skill(
                        Skill(
                            activation=f"worker skill {offset}-{i}",
                            execution="do stuff",
                            termination="done",
                        ),
                        embedding=[offset, offset + 0.1, offset + 0.2, float(i) * 0.01],
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=insert, args=(5, 0.1))
        t2 = threading.Thread(target=insert, args=(5, 0.5))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All inserts should have succeeded.
        assert errors == [], f"concurrency errors: {errors}"
        assert store.count_skills() == 10

    def test_concurrent_reads_during_writes(self, tmp_path: Path) -> None:
        """Readers should not see partial state or raise."""
        db_path = tmp_path / "rw.db"
        store = Store(db_path, embedding_dims=4)

        # seed one skill
        store.add_skill(
            Skill(activation="initial", execution="x", termination="y"),
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        read_errors: list[Exception] = []
        stop = False

        def reader() -> None:
            reader_store = Store(db_path, embedding_dims=4)
            while not stop:
                try:
                    reader_store.list_skills()
                    reader_store.search_skills_by_embedding(
                        [0.1, 0.2, 0.3, 0.4], top_k=3
                    )
                except Exception as e:  # noqa: BLE001
                    read_errors.append(e)
                time.sleep(0.001)

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        # meanwhile, insert a bunch of skills on the main thread
        writer_store = Store(db_path, embedding_dims=4)
        for i in range(20):
            writer_store.add_skill(
                Skill(
                    activation=f"skill {i}",
                    execution="x",
                    termination="y",
                ),
                embedding=[0.1 * i, 0.2, 0.3, 0.4],
            )

        stop = True
        reader_thread.join(timeout=2)
        assert read_errors == [], f"reads crashed: {read_errors}"


# ----------------------------------------------------------------------
# Retrieval performance / scale
# ----------------------------------------------------------------------


class TestRetrievalScale:
    def test_retrieval_latency_with_many_skills(self, tmp_path: Path) -> None:
        """At 500 skills, KNN search should still be snappy."""
        db_path = tmp_path / "big.db"
        store = Store(db_path, embedding_dims=16)

        # Pre-build 500 skills with randomized embeddings
        import random

        rng = random.Random(42)
        for i in range(500):
            emb = [rng.uniform(-1, 1) for _ in range(16)]
            # normalize
            norm = sum(x * x for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            store.add_skill(
                Skill(
                    activation=f"skill number {i} for various tasks",
                    execution="do the thing",
                    termination="done",
                ),
                embedding=emb,
            )
        assert store.count_skills() == 500

        # Time a single retrieval
        query = [rng.uniform(-1, 1) for _ in range(16)]
        norm = sum(x * x for x in query) ** 0.5
        query = [x / norm for x in query]

        start = time.perf_counter()
        hits = store.search_skills_by_embedding(query, top_k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(hits) == 5
        # With sqlite-vec at 500 skills, KNN should be well under 100ms.
        # We assert a generous 500ms budget so CI environments don't flake.
        assert elapsed_ms < 500, f"retrieval took {elapsed_ms:.1f}ms at 500 skills"


# ----------------------------------------------------------------------
# Outcome heuristic edge cases
# ----------------------------------------------------------------------


class TestOutcomeEdgeCases:
    def test_trajectory_with_no_output_is_unknown(self) -> None:
        traj = Trajectory(
            tool_calls=[
                ToolCall(
                    name="Bash",
                    arguments={"command": "true"},
                    result="",  # empty output, no error
                )
            ]
        )
        sig = infer_outcome(traj)
        assert sig.outcome is Outcome.UNKNOWN

    def test_mixed_errors_and_recovery_is_success(self) -> None:
        """Mid-trajectory errors that get recovered from should still count
        as success if the final state is clean."""
        traj = Trajectory(
            tool_calls=[
                ToolCall(name="Bash", arguments={"command": "x"}, error="first try failed"),
                ToolCall(name="Bash", arguments={"command": "x --retry"}, error="still bad"),
                ToolCall(
                    name="Bash",
                    arguments={"command": "x --force"},
                    result="/path/to/output/__init__.py",
                ),
            ]
        )
        sig = infer_outcome(traj, any_skills_activated=True)
        # final state has __init__.py success marker → SUCCESS despite
        # 2 errors earlier in the trajectory
        assert sig.outcome is Outcome.SUCCESS

    def test_failure_keywords_in_body_not_final_output(self) -> None:
        """Error words in middle outputs shouldn't override a clean final."""
        traj = Trajectory(
            tool_calls=[
                ToolCall(
                    name="Bash",
                    arguments={"command": "run"},
                    result="Traceback (most recent call last) lots of error stuff",
                ),
                ToolCall(
                    name="Bash",
                    arguments={"command": "fix"},
                    result="build successful",
                ),
            ]
        )
        sig = infer_outcome(traj)
        assert sig.outcome is Outcome.SUCCESS

    def test_permission_denied_is_failure(self) -> None:
        traj = Trajectory(
            tool_calls=[
                ToolCall(
                    name="Bash",
                    arguments={"command": "cat /etc/shadow"},
                    result="cat: /etc/shadow: Permission denied",
                )
            ]
        )
        sig = infer_outcome(traj)
        assert sig.outcome is Outcome.FAILURE
