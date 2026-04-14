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
    assert "list" in out
    assert "stats" in out
    assert "bootstrap" in out
    assert "refine" in out
    assert "retrieve" in out
    assert "ingest" in out
    assert "maint" in out
    assert "share" in out
    assert "review" in out
    assert "eval" in out

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
