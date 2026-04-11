"""Tests for release note extraction."""

from __future__ import annotations

import pytest

from muscle_memory.release_notes import extract_release_notes


def test_extract_release_notes_returns_one_section() -> None:
    changelog = """# Changelog

## [0.7.3] — 2026-04-11

### Added

- New release workflow

## [0.7.2] — 2026-04-10

- Older notes
"""

    notes = extract_release_notes("0.7.3", changelog)
    assert "## [0.7.3]" in notes
    assert "New release workflow" in notes
    assert "## [0.7.2]" not in notes


def test_extract_release_notes_raises_for_missing_version() -> None:
    with pytest.raises(ValueError, match=r"CHANGELOG\.md does not contain ## \[0.9.0\]"):
        extract_release_notes("0.9.0", "# Changelog\n")
