"""Tests for release workflow supply-chain hardening."""

from __future__ import annotations

from pathlib import Path

RELEASE_WORKFLOW = Path(".github/workflows/release.yml")
DEVELOPMENT_DOC = Path("docs/development.md")


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
