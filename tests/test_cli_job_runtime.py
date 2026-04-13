from __future__ import annotations

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
