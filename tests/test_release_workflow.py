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


def test_release_workflow_skips_attestations_for_private_repos() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "if: ${{ !github.event.repository.private }}" in text


def test_release_workflow_builds_and_uploads_binaries() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "build-binaries:" in text
    assert "macos-latest" in text
    assert "ubuntu-latest" in text
    assert "scripts/build_release_binaries.py" in text
    assert "scripts/check_release_binaries.py" in text
    assert "actions/upload-artifact@v4" in text
    assert "actions/download-artifact@v4" in text
    assert "install.sh" in text


def test_release_workflow_makes_pypi_optional_and_non_blocking() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "default: false" in text
    assert "publish-pypi:" in text
    assert "continue-on-error: true" in text
    assert "if: ${{ inputs.publish_to_pypi }}" in text


def test_development_docs_note_release_attestations() -> None:
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")

    assert "artifact attestations" in text.lower()


def test_release_workflow_mentions_release_checklist_and_gate() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "docs/release.md" in text
    assert "release checklist" in text.lower()
    assert "scripts/release_benchmark_gate.py" in text
    assert "tests/test_behavioral.py" in text
    assert 'CLAUDE_TESTS: "1"' in text


def test_release_workflow_supports_rerun_when_tag_already_exists_on_same_commit() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert 'git rev-list -n 1 "refs/tags/v${VERSION}"' in text
    assert "reusing it for rerun" in text


def test_release_checklist_doc_exists_and_lists_gate_steps() -> None:
    text = RELEASE_DOC.read_text(encoding="utf-8")

    assert "# Release Checklist" in text
    assert "release_benchmark_gate.py" in text
    assert "release_preflight.py" in text
    assert "build_release_binaries.py" in text
    assert "install.sh" in text
    assert "SHA256SUMS" in text


def test_testing_and_demo_docs_note_supported_surface_and_recovery() -> None:
    testing_text = TESTING_DOC.read_text(encoding="utf-8")
    demo_text = DEMO_DOC.read_text(encoding="utf-8")

    assert "Claude Code still has the deepest runtime coverage" in testing_text
    assert "Codex is" in testing_text
    assert "recovery" in testing_text.lower()
    assert "deepest runtime integration" in demo_text
    assert "Codex" in demo_text
    assert "mm status" in demo_text
    assert "mm doctor" in demo_text
    assert "mm skills" in demo_text


def test_development_docs_distinguish_preflight_from_full_ci_path() -> None:
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")

    assert "first release gate" in text.lower()
    assert "full CI-equivalent release path" in text
    assert "build_release_binaries.py" in text
    assert "install.sh" in text
    assert "docs/release.md" in text
