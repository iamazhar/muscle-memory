"""Tests for CI workflow release gate coverage."""

from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")
DEVELOPMENT_DOC = Path("docs/development.md")
README = Path("README.md")


def test_ci_package_job_runs_release_preflight() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "scripts/release_preflight.py" in text


def test_ci_runs_eval_benchmark_gate() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "mm eval run --json" in text
    assert "thresholds_passed" in text
    assert "failed_thresholds" in text


def test_development_docs_note_release_preflight() -> None:
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")

    assert "release preflight" in text.lower()


def test_readme_calls_out_claude_code_first_release_story() -> None:
    text = README.read_text(encoding="utf-8")

    assert "Claude Code-first" in text
    assert "mm doctor" in text
    assert "mm maint pause" in text
