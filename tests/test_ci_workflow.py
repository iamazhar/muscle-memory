"""Tests for CI workflow release preflight coverage."""

from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")
DEVELOPMENT_DOC = Path("docs/development.md")


def test_ci_package_job_runs_release_preflight() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "scripts/release_preflight.py" in text


def test_development_docs_note_release_preflight() -> None:
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")

    assert "release preflight" in text.lower()
