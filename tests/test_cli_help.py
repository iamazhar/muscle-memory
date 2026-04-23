"""Tests for the public CLI surface and help output."""

from __future__ import annotations

from typer.testing import CliRunner

from muscle_memory.cli import app

runner = CliRunner()


def test_top_level_help_is_trimmed() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    out = result.output
    assert "init" in out
    assert "learn" in out
    assert "retrieve" in out
    assert "skills" in out
    assert "show" in out
    assert "status" in out
    assert "doctor" in out

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


def test_version_flag_is_available() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "muscle-memory" in result.output


def test_maint_help_lists_maintenance_commands() -> None:
    result = runner.invoke(app, ["maint", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "pause" in out
    assert "resume" in out
    assert "dedup" in out
    assert "rescore" in out
    assert "prune" in out
    assert "govern" in out


def test_share_help_lists_transfer_commands() -> None:
    result = runner.invoke(app, ["share", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "export" in out
    assert "import" in out


def test_review_help_lists_review_commands() -> None:
    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "list" in out
    assert "approve" in out
    assert "reject" in out


def test_jobs_help_lists_job_commands() -> None:
    result = runner.invoke(app, ["jobs", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "list" in out
    assert "retry" in out
