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


def _make_config(store_dir: Path, *, debug_enabled: bool = False) -> Config:
    return Config(
        db_path=store_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=store_dir,
        debug_enabled=debug_enabled,
    )


def test_doctor_handles_missing_db(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, debug_enabled=False)

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["db_exists"] is False
    assert data["job_counts"]["pending"] == 0
    assert data["debug_enabled"] is False


def test_doctor_reports_jobs_and_debug_log(tmp_path: Path) -> None:
    store_dir = tmp_path
    claude_dir = store_dir / ".claude"
    claude_dir.mkdir()
    cfg = _make_config(store_dir, debug_enabled=True)
    store = Store(cfg.db_path)
    store.add_job(BackgroundJob(kind=JobKind.EXTRACT, payload={}, status=JobStatus.PENDING))
    store.add_job(
        BackgroundJob(kind=JobKind.REFINE, payload={}, status=JobStatus.FAILED, error="boom")
    )
    log_path = claude_dir / "mm.debug.log"
    log_path.write_text(
        '{"event":"spawned_extraction","component":"stop"}\n'
        '{"event":"no_hits","component":"user_prompt","session_id":"sess-a","prompt_excerpt":"capital of france","lexical_prefilter_skipped":false,"reject_reason":"weak_match_without_lexical_support","retrieve_ms":0.2,"total_ms":0.2}\n'
        '{"event":"hits_returned","component":"user_prompt","session_id":"sess-b","prompt_excerpt":"debug pth","retrieve_ms":12.5,"embed_ms":4.0,"search_ms":1.0,"rerank_ms":0.5,"total_ms":13.0,"hit_count":1}\n',
        encoding="utf-8",
    )

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["db_exists"] is True
    assert data["debug_enabled"] is True
    assert data["debug_log_exists"] is True
    assert data["job_counts"]["pending"] == 1
    assert data["job_counts"]["failed"] == 1
    assert data["last_debug_event"]["event"] == "hits_returned"
    assert data["retrieval_telemetry"]["samples"] == 2
    assert data["retrieval_telemetry"]["avg_total_ms"] == 6.6
    assert len(data["recent_retrieval_decisions"]) == 2
    assert data["recent_retrieval_decisions"][0]["event"] == "hits_returned"
    assert (
        data["recent_retrieval_decisions"][1]["why"]
        == "weak semantic hit without lexical corroboration"
    )


def test_doctor_maps_lexical_prefilter_skip_reason(tmp_path: Path) -> None:
    store_dir = tmp_path
    claude_dir = store_dir / ".claude"
    claude_dir.mkdir()
    cfg = _make_config(store_dir, debug_enabled=True)
    Store(cfg.db_path)
    log_path = claude_dir / "mm.debug.log"
    log_path.write_text(
        '{"event":"no_hits","component":"user_prompt","session_id":"sess-a","prompt_excerpt":"capital of france","lexical_prefilter_skipped":true,"retrieve_ms":0.2,"total_ms":0.2}\n',
        encoding="utf-8",
    )

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["recent_retrieval_decisions"]) == 1
    assert data["recent_retrieval_decisions"][0]["why"] == "no lexical overlap with trusted skills"


def test_doctor_maps_similarity_floor_reject_reason(tmp_path: Path) -> None:
    store_dir = tmp_path
    claude_dir = store_dir / ".claude"
    claude_dir.mkdir()
    cfg = _make_config(store_dir, debug_enabled=True)
    Store(cfg.db_path)
    log_path = claude_dir / "mm.debug.log"
    log_path.write_text(
        '{"event":"no_hits","component":"user_prompt","session_id":"sess-a","prompt_excerpt":"capital of france","reject_reason":"distance_below_similarity_floor","retrieve_ms":0.2,"total_ms":0.2}\n',
        encoding="utf-8",
    )

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["recent_retrieval_decisions"]) == 1
    assert (
        data["recent_retrieval_decisions"][0]["why"]
        == "semantic match fell below the configured similarity floor"
    )


def test_doctor_recommendations_include_debug_and_resume(tmp_path: Path) -> None:
    store_dir = tmp_path
    claude_dir = store_dir / ".claude"
    claude_dir.mkdir()
    cfg = _make_config(store_dir, debug_enabled=False)
    Store(cfg.db_path)
    (claude_dir / "mm.paused").touch()

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["recommendations"] == [
        "Enable MM_DEBUG=1 while validating Claude Code retrieval decisions.",
        "Run advanced `mm maint resume` before dogfooding if the project is paused.",
    ]

    with patch("muscle_memory.cli._load_config", return_value=cfg):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Enable MM_DEBUG=1 while validating Claude Code retrieval decisions." in result.output
    assert (
        "Run advanced `mm maint resume` before dogfooding if the project is paused."
        in result.output
    )
