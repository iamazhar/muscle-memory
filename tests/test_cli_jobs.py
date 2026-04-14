from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from muscle_memory.cli import app
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import BackgroundJob, JobKind, JobStatus, Scope

runner = CliRunner()


def _make_config(store_dir: Path) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
    )


def test_jobs_list_outputs_json(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)
    job = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "ep-1"})
    store.add_job(job)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
    ):
        result = runner.invoke(app, ["jobs", "list", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["id"] == job.id
    assert data[0]["kind"] == "extract"


def test_jobs_retry_failed_extract_requeues_job(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)
    job = BackgroundJob(
        kind=JobKind.EXTRACT,
        payload={"episode_id": "ep-1"},
        status=JobStatus.FAILED,
        error="boom",
    )
    store.add_job(job)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli._spawn_job_retry") as spawn_retry,
    ):
        result = runner.invoke(app, ["jobs", "retry", job.id])

    assert result.exit_code == 0
    spawn_retry.assert_called_once()
    updated = store.get_job(job.id)
    assert updated is not None
    assert updated.status is JobStatus.PENDING
    assert updated.error is None


def test_jobs_retry_failed_requeues_all_failed_jobs(tmp_path: Path) -> None:
    store_dir = tmp_path
    (store_dir / ".claude").mkdir()
    cfg = _make_config(store_dir)
    store = Store(cfg.db_path)
    failed_a = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "ep-1"}, status=JobStatus.FAILED, error="boom")
    failed_b = BackgroundJob(kind=JobKind.REFINE, payload={}, status=JobStatus.FAILED, error="oops")
    ok = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": "ep-2"}, status=JobStatus.SUCCEEDED)
    store.add_job(failed_a)
    store.add_job(failed_b)
    store.add_job(ok)

    with (
        patch("muscle_memory.cli._load_config", return_value=cfg),
        patch("muscle_memory.cli._open_store", return_value=store),
        patch("muscle_memory.cli._spawn_job_retry") as spawn_retry,
    ):
        result = runner.invoke(app, ["jobs", "retry-failed"])

    assert result.exit_code == 0
    assert spawn_retry.call_count == 2
    assert store.get_job(failed_a.id).status is JobStatus.PENDING
    assert store.get_job(failed_b.id).status is JobStatus.PENDING
    assert store.get_job(ok.id).status is JobStatus.SUCCEEDED
