"""Tests for CI workflow release gate coverage."""

from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")
DEVELOPMENT_DOC = Path("docs/development.md")
README = Path("README.md")


def test_ci_package_job_runs_release_preflight() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "scripts/release_preflight.py" in text


def test_ci_runs_release_benchmark_gate() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "scripts/release_benchmark_gate.py" in text
    assert "thresholds_passed" in text
    assert "failed_thresholds" in text


def test_development_docs_note_release_preflight() -> None:
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")

    assert "release preflight" in text.lower()


def test_readme_calls_out_supported_harnesses_and_recovery() -> None:
    text = README.read_text(encoding="utf-8")

    assert "Claude Code has the deepest runtime integration today" in text
    assert "Codex is also supported" in text
    assert "mm doctor" in text
    assert "mm maint pause" in text
    assert "mm review list" in text
    assert "mm jobs retry-failed" in text
