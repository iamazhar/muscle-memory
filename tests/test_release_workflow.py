"""Tests for release workflow supply-chain hardening."""

from __future__ import annotations

from pathlib import Path

RELEASE_WORKFLOW = Path(".github/workflows/release.yml")
DEVELOPMENT_DOC = Path("docs/development.md")
RELEASE_DOC = Path("docs/release.md")
DEMO_DOC = Path("docs/demo.md")
TESTING_DOC = Path("docs/testing.md")


def test_release_workflow_grants_attestation_permissions() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "attestations: write" in text
    assert "id-token: write" in text


def test_release_workflow_attests_release_artifacts() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "uses: actions/attest@v4" in text
    assert "subject-path: |" in text
    assert "dist/*.whl" in text
    assert "dist/*.tar.gz" in text


def test_release_workflow_attests_before_tag_and_release_creation() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    attest_index = text.index("- name: Generate artifact attestations")
    tag_index = text.index("- name: Create and push tag")
    release_index = text.index("- name: Create GitHub release")

    assert attest_index < tag_index
    assert attest_index < release_index


def test_development_docs_note_release_attestations() -> None:
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")

    assert "artifact attestations" in text.lower()


def test_release_workflow_mentions_release_checklist_and_gate() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "docs/release.md" in text
    assert "release checklist" in text.lower()
    assert "mm eval run --json" in text


def test_release_checklist_doc_exists_and_lists_gate_steps() -> None:
    text = RELEASE_DOC.read_text(encoding="utf-8")

    assert "# Release Checklist" in text
    assert "mm eval run --json" in text
    assert "release_preflight.py" in text


def test_testing_and_demo_docs_note_supported_surface_and_recovery() -> None:
    testing_text = TESTING_DOC.read_text(encoding="utf-8")
    demo_text = DEMO_DOC.read_text(encoding="utf-8")

    assert "Claude Code-first" in testing_text
    assert "recovery" in testing_text.lower()
    assert "Claude Code-first" in demo_text
    assert "mm doctor" in demo_text
