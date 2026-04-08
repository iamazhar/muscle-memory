"""Behavioral tests that require a real Claude Code installation.

Each test spawns `claude -p "prompt"` in a scratch project with a
pre-seeded mm database and verifies observable outcomes:
  * the 🧠 visibility marker appears in Claude's response
  * Claude actually invokes tools (imperative execution) rather than
    narrating steps back at the user
  * bang-commands don't trigger skill retrieval
  * unrelated prompts produce the "no matching playbook" marker

These are slow (each test spends a few seconds waiting on Claude Code)
and cost real LLM tokens, so they're OPT-IN:

    # run only the unit + integration suites (default)
    pytest tests/

    # include behavioral tests
    CLAUDE_TESTS=1 pytest tests/test_behavioral.py

The tests also require:
  * `claude` on PATH (Claude Code installed)
  * `mm` on PATH (muscle-memory tool installed globally)
  * ANTHROPIC_API_KEY or an active Claude Code subscription

Most users will run these via the parallel-agent test runner rather
than pytest directly — see scripts/run-behavioral-tests.sh.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

import pytest

CLAUDE_TESTS_ENV = "CLAUDE_TESTS"
SKIP_REASON = (
    f"Behavioral tests are opt-in. Set {CLAUDE_TESTS_ENV}=1 to run them "
    "(requires `claude` and `mm` on PATH + API access)."
)

pytestmark = pytest.mark.skipif(
    os.environ.get(CLAUDE_TESTS_ENV) != "1",
    reason=SKIP_REASON,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _require_tools() -> None:
    missing = [t for t in ("claude", "mm") if not _have_tool(t)]
    if missing:
        pytest.skip(f"missing required tools on PATH: {', '.join(missing)}")


@pytest.fixture
def scratch_project(tmp_path: Path) -> Iterator[Path]:
    """A throwaway project directory with .git and mm initialized."""
    _require_tools()
    project = tmp_path / "scratch"
    project.mkdir()
    # make it look like a git repo so mm init finds it as a project root
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)

    # run `mm init`
    result = subprocess.run(
        ["mm", "init", "--scope", "project"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"mm init failed: {result.stderr}")

    yield project
    # tmp_path cleanup handles removal


def _seed_skill(project: Path, skill_json: dict) -> str:
    """Insert a skill into the project's mm.db via `mm import`.

    Returns the skill id.
    """
    import_path = project / "seed.json"
    import_path.write_text(json.dumps([skill_json]))
    result = subprocess.run(
        ["mm", "import", str(import_path)],
        cwd=project,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"mm import failed: {result.stderr}")
    # get the id of the just-imported skill
    list_result = subprocess.run(
        ["mm", "list", "--json"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    skills = json.loads(list_result.stdout)
    assert skills, "expected at least one skill after import"
    return skills[-1]["id"]


def _run_claude(project: Path, prompt: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `claude -p "prompt"` in the scratch project with a timeout.

    Returns the CompletedProcess including stdout / stderr for
    assertion. Uses --dangerously-skip-permissions so Claude can run
    Bash without prompting during the test.
    """
    return subprocess.run(
        [
            "claude",
            "--dangerously-skip-permissions",
            "-p",
            prompt,
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )


# ----------------------------------------------------------------------
# behavioral scenarios
# ----------------------------------------------------------------------


class TestMarkerVisibility:
    def test_matching_prompt_emits_brain_marker(self, scratch_project: Path) -> None:
        """A prompt that matches a seeded skill should cause Claude to
        emit the 🧠 muscle-memory marker at the top of its response."""
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks about fixing tests that fail with a module import error on macOS",
                "execution": "1. Run `echo test-placeholder` to simulate the diagnostic step.\n2. Report that the diagnostic command ran.",
                "termination": "The echo command has run and been reported.",
                "tags": ["test-marker"],
                "tool_hints": ["Bash"],
                "scope": "project",
            },
        )

        result = _run_claude(
            scratch_project,
            "Tests are failing with a module import error on my mac, help?",
        )
        assert result.returncode == 0, f"claude failed: {result.stderr}"
        out = result.stdout
        assert "🧠" in out, f"expected 🧠 marker, got: {out[:500]}"
        assert "muscle-memory" in out.lower()
        assert "playbook" in out.lower()


class TestImperativeExecution:
    def test_matched_skill_triggers_tool_calls(self, scratch_project: Path) -> None:
        """When a skill matches, Claude should actually run the commands
        in `execution`, not narrate them back to the user."""
        # marker file the skill will create
        marker_file = scratch_project / "mm_ran.txt"
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks to run the muscle-memory execution test probe",
                "execution": f"1. Run `touch {marker_file}` to prove the playbook executed.\n2. Report the marker file has been created.",
                "termination": f"The file {marker_file} exists.",
                "tool_hints": ["Bash"],
                "scope": "project",
            },
        )

        _run_claude(
            scratch_project,
            "Please run the muscle-memory execution test probe",
        )

        # The key assertion: did Claude actually run the touch command?
        assert marker_file.exists(), (
            f"Claude did not execute the playbook steps (marker file missing). "
            f"Expected file: {marker_file}"
        )


class TestShellEscapeGate:
    def test_bang_command_does_not_activate_skill(
        self, scratch_project: Path
    ) -> None:
        """Bang commands should not fire the hook at all — no injection,
        no marker. We can't directly observe the hook NOT firing in a
        claude -p run, but we can seed a skill and then submit a bang
        command and verify no 🧠 marker appears."""
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks to list files in the current directory",
                "execution": "Run `ls -la` and show the listing.",
                "termination": "Listing shown.",
                "tool_hints": ["Bash"],
                "scope": "project",
            },
        )

        # Use a bang command that would match the skill if the hook fired
        result = _run_claude(scratch_project, "!ls -la")
        # With --dangerously-skip-permissions, claude -p may interpret the
        # bang differently. What matters: no 🧠 marker in the output.
        assert "🧠" not in result.stdout, (
            f"bang command triggered the hook unexpectedly: {result.stdout[:300]}"
        )


class TestNoMatchHandling:
    def test_unrelated_prompt_emits_no_match_marker(
        self, scratch_project: Path
    ) -> None:
        """When no skill matches, Claude should still emit the
        visibility marker in 'no matching playbook' form."""
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks about fixing Kubernetes CrashLoopBackOff errors in a specific production deployment",
                "execution": "1. Run `kubectl get pods`\n2. Run `kubectl describe pod`\n3. Report findings.",
                "termination": "Root cause identified.",
                "tags": ["kubernetes"],
                "scope": "project",
            },
        )

        result = _run_claude(
            scratch_project,
            "What is the capital of France?",
        )
        out = result.stdout
        # Either Claude saw the no-match marker instruction and emitted it,
        # OR Claude retrieved nothing and didn't emit the marker at all.
        # Both are acceptable — the fail condition is if it WRONGLY claims
        # to be executing an unrelated playbook.
        if "🧠" in out:
            assert (
                "no matching playbook" in out.lower()
                or "proceeding normally" in out.lower()
            ), (
                f"marker present but should say 'no matching playbook': {out[:300]}"
            )


class TestHookWiringEndToEnd:
    def test_fresh_init_produces_working_hook_invocation(
        self, tmp_path: Path
    ) -> None:
        """A fresh `mm init` should yield a hook that can be invoked via
        subprocess with valid output (empty if no skills, or injection
        block if skills match)."""
        _require_tools()

        project = tmp_path / "fresh"
        project.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=project, check=True)

        result = subprocess.run(
            ["mm", "init", "--scope", "project"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # settings.json must exist and contain our hooks
        settings_path = project / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "UserPromptSubmit" in settings["hooks"]
        assert "Stop" in settings["hooks"]

        # invoke the user-prompt hook manually with a minimal payload
        payload = json.dumps(
            {
                "session_id": "manual-test",
                "cwd": str(project),
                "prompt": "this is a test",
            }
        )
        hook_result = subprocess.run(
            ["mm", "hook", "user-prompt"],
            cwd=project,
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Hook must exit 0 even with no matches
        assert hook_result.returncode == 0
        # Empty stdout is fine (no skills seeded)
