from __future__ import annotations

import json
from pathlib import Path
from io import StringIO
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import BackgroundJob, JobKind, JobStatus, Maturity, Scope, Skill
from muscle_memory.hooks.stop import main as stop_main

runner = CliRunner()


def _make_config(store_dir: Path) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
    )


def test_extract_episode_marks_job_failed_when_episode_missing(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)
    job = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "missing"}, status=JobStatus.RUNNING)
    store.add_job(job)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["extract-episode", "missing", "--job-id", job.id])

    assert result.exit_code == 1
    updated = store.get_job(job.id)
    assert updated is not None
    assert updated.status is JobStatus.FAILED
    assert updated.error


def test_stop_hook_skips_refinement_when_auto_refine_disabled(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
        auto_refine_enabled=False,
    )
    store = Store(cfg.db_path)
    store.add_skill(
        Skill(
            activation="When tests fail",
            execution="1. Run pytest\n2. Fix the failure",
            termination="Tests pass",
            maturity=Maturity.LIVE,
            invocations=8,
            successes=2,
            failures=6,
            score=0.25,
        )
    )
    transcript = store_dir / "sess.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "run the tests"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "OK"},
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
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {"session_id": "sess-1", "cwd": str(store_dir), "transcript_path": str(transcript)}
    stdin = StringIO(json.dumps(payload))

    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.stop.subprocess.Popen"),
        patch("sys.stdin", stdin),
        patch("sys.stdout", StringIO()),
        patch("muscle_memory.hooks.stop._fire_async_refinement") as fire_refinement,
    ):
        rc = stop_main()

    assert rc == 0
    fire_refinement.assert_not_called()
    refine_jobs = store.list_jobs(limit=None, kind=JobKind.REFINE)
    assert refine_jobs == []


def test_stop_hook_skips_refinement_when_refine_job_is_active(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)
    store.add_skill(
        Skill(
            activation="When tests fail",
            execution="1. Run pytest\n2. Fix the failure",
            termination="Tests pass",
            maturity=Maturity.LIVE,
            invocations=8,
            successes=2,
            failures=6,
            score=0.25,
        )
    )
    store.add_job(BackgroundJob(kind=JobKind.REFINE, payload={}, status=JobStatus.RUNNING))
    transcript = store_dir / "sess.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "run the tests"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "OK"},
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
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {"session_id": "sess-1", "cwd": str(store_dir), "transcript_path": str(transcript)}
    stdin = StringIO(json.dumps(payload))

    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.stop.subprocess.Popen"),
        patch("sys.stdin", stdin),
        patch("sys.stdout", StringIO()),
        patch("muscle_memory.hooks.stop._fire_async_refinement") as fire_refinement,
    ):
        rc = stop_main()

    assert rc == 0
    fire_refinement.assert_not_called()
    refine_jobs = store.list_jobs(limit=None, kind=JobKind.REFINE)
    assert len(refine_jobs) == 1
    assert refine_jobs[0].status is JobStatus.RUNNING
