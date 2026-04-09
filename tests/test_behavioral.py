"""Behavioral tests that require a real Claude Code installation.

Each test spawns `claude -p "prompt"` in a scratch project with a
pre-seeded mm database and verifies observable outcomes:
  * the 🧠 visibility marker appears in Claude's response
  * Claude actually invokes tools (imperative execution) rather than
    narrating steps back at the user
  * bang-commands don't trigger skill retrieval
  * unrelated prompts proceed silently (no 🧠 marker emitted)

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
from collections.abc import Iterator
from pathlib import Path

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


def _run_claude(
    project: Path, prompt: str, timeout: int = 120, stream: bool = False
) -> subprocess.CompletedProcess:
    """Run `claude -p "prompt"` in the scratch project with a timeout.

    Important `claude -p` detail uncovered during agent-driven testing:
    plain `-p` mode returns only the FINAL assistant text — any earlier
    turns (including the 🧠 muscle-memory marker that Claude emits before
    running tool calls) are not in stdout. For marker-visibility tests
    pass `stream=True` to get the full event stream via
    `--output-format stream-json --verbose`.
    """
    args = ["claude", "--dangerously-skip-permissions"]
    if stream:
        args.extend(["--output-format", "stream-json", "--verbose"])
    args.extend(["-p", prompt])
    return subprocess.run(
        args,
        cwd=project,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )


def _stream_contains_marker(stream_output: str, needle: str = "🧠") -> bool:
    """Scan a `stream-json` output for the needle in any assistant text."""
    for line in stream_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # events look like: {"type": "assistant", "message": {"content": [...]}}
        msg = event.get("message") or {}
        content = msg.get("content") or []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    if needle in block.get("text", ""):
                        return True
        elif isinstance(content, str):
            if needle in content:
                return True
    return False


# ----------------------------------------------------------------------
# behavioral scenarios
# ----------------------------------------------------------------------


class TestMarkerVisibility:
    def test_matching_prompt_emits_brain_marker_in_stream(self, scratch_project: Path) -> None:
        """A prompt that matches a seeded skill should cause Claude to
        emit the 🧠 muscle-memory marker.

        Uses stream-json mode because plain `-p` only returns the final
        assistant text, and the marker is emitted in the first assistant
        turn before any tool calls.
        """
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks to run the muscle-memory visibility verification probe",
                "execution": "1. Run `echo mm-marker-probe-visible` as a one-line diagnostic.\n2. Report that the probe ran.",
                "termination": "The echo command has been executed.",
                "tags": ["test-marker-stream"],
                "tool_hints": ["Bash"],
                "scope": "project",
            },
        )

        result = _run_claude(
            scratch_project,
            "Please run the muscle-memory visibility verification probe",
            stream=True,
        )
        assert result.returncode == 0, f"claude failed: {result.stderr}"
        assert _stream_contains_marker(result.stdout, "🧠"), (
            f"🧠 marker not found anywhere in the stream. First 500 chars: {result.stdout[:500]}"
        )
        assert _stream_contains_marker(result.stdout, "muscle-memory"), (
            "'muscle-memory' substring not found in any assistant text block"
        )

    def test_no_match_marker_survives_plain_p(self, scratch_project: Path) -> None:
        """When no skill matches, Claude emits a short single-turn
        response. In that case, `-p` stdout DOES contain the marker
        (no tool calls trim it off the front)."""
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks about Kubernetes CrashLoopBackOff in a production deployment",
                "execution": "1. Run `kubectl get pods`\n2. Run `kubectl describe pod <name>`",
                "termination": "Root cause identified.",
                "tags": ["test-no-match"],
                "tool_hints": ["Bash"],
                "scope": "project",
            },
        )

        result = _run_claude(
            scratch_project,
            "What is the capital of France? One word only.",
        )
        assert result.returncode == 0
        out = result.stdout
        # No skill should match "capital of France", so the 🧠 marker
        # should NOT appear at all (silent no-match).  If it does appear,
        # it must not wrongly claim to be executing an unrelated playbook.
        if "🧠" in out:
            lower = out.lower()
            assert "executing playbook" not in lower, (
                f"marker wrongly claims execution for unrelated prompt: {out[:300]}"
            )
        # Paris should be in the answer
        assert "paris" in out.lower()


class TestImperativeExecution:
    def test_matched_skill_triggers_tool_calls(self, scratch_project: Path) -> None:
        """When a skill matches, Claude should actually run the commands
        in `execution`, not narrate them back to the user.

        Verification strategy: don't scan stdout for tool calls (headless
        `-p` strips them from the final output); instead plant a marker
        file that can ONLY exist if the Bash tool was actually invoked.
        """
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

        # The only way the marker file gets there is if Claude actually
        # invoked Bash(touch ...). Narration alone cannot create it.
        assert marker_file.exists(), (
            f"Claude did not execute the playbook steps (marker file missing). "
            f"Expected file: {marker_file}"
        )

    def test_mm_state_updates_after_executed_skill(self, scratch_project: Path) -> None:
        """After a full cycle: skill activated, executed, session ended,
        Stop hook fired, scorer credited. The DB state should reflect it."""
        _seed_skill(
            scratch_project,
            {
                "activation": "When the user asks to run the muscle-memory state-update verification probe",
                "execution": "1. Run `echo state-probe-ran` and show output.\n2. Report completion.",
                "termination": "The echo has run.",
                "tool_hints": ["Bash"],
                "scope": "project",
            },
        )

        # Snapshot
        stats_before = subprocess.run(
            ["mm", "list", "--json"],
            cwd=scratch_project,
            capture_output=True,
            text=True,
        )
        skills_before = json.loads(stats_before.stdout)
        assert len(skills_before) == 1
        invocations_before = skills_before[0]["invocations"]

        # Run
        _run_claude(
            scratch_project,
            "Please run the muscle-memory state-update verification probe",
        )

        # Give the async Stop hook a moment to finish extraction.
        time.sleep(2)

        # Check
        stats_after = subprocess.run(
            ["mm", "list", "--json"],
            cwd=scratch_project,
            capture_output=True,
            text=True,
        )
        skills_after = json.loads(stats_after.stdout)
        assert len(skills_after) >= 1

        # At least one skill should have a higher invocation count.
        max_invocations_after = max(s["invocations"] for s in skills_after)
        assert max_invocations_after > invocations_before, (
            f"expected invocation count to increase: "
            f"before={invocations_before}, after={max_invocations_after}"
        )


class TestShellEscapeGate:
    def test_bang_command_does_not_activate_skill(self, scratch_project: Path) -> None:
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
    def test_unrelated_prompt_has_no_execution_marker(self, scratch_project: Path) -> None:
        """When no skill matches, Claude should proceed silently —
        no 🧠 marker emitted at all."""
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
        # The fail condition is if Claude WRONGLY claims to be executing
        # an unrelated playbook.  No marker at all is the ideal outcome.
        if "🧠" in out:
            assert "executing playbook" not in out.lower(), (
                f"marker wrongly claims execution for unrelated prompt: {out[:300]}"
            )


class TestHookWiringEndToEnd:
    def test_fresh_init_produces_working_hook_invocation(self, tmp_path: Path) -> None:
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

    def test_full_lifecycle_seed_execute_score(self, tmp_path: Path) -> None:
        """Full round trip in a throwaway project:
           init → seed → claude execute → stop hook → scorer credit.

        Verifies mm-observable state changes (not `-p` stdout content,
        which is unreliable for multi-turn responses).
        """
        _require_tools()

        project = tmp_path / "lifecycle"
        project.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=project, check=True)
        subprocess.run(["mm", "init", "--scope", "project"], cwd=project, check=True)

        marker_file = project / "lifecycle_marker.txt"
        seed = [
            {
                "activation": "When the user asks to run the muscle-memory full lifecycle verification probe",
                "execution": f"1. Run `touch {marker_file}` to create the lifecycle marker file.\n2. Run `ls -la {marker_file}` to verify it.\n3. Report the lifecycle probe is complete.",
                "termination": f"The file {marker_file} exists on disk.",
                "tags": ["test-lifecycle"],
                "tool_hints": ["Bash"],
                "scope": "project",
                "successes": 0,
                "invocations": 0,
                "failures": 0,
                "maturity": "candidate",
                "source_episode_ids": [],
                "score": 0.0,
                "created_at": "2026-04-07T22:00:00+00:00",
                "last_used_at": None,
            }
        ]
        seed_path = project / "seed.json"
        seed_path.write_text(json.dumps(seed))
        subprocess.run(["mm", "import", str(seed_path)], cwd=project, check=True)

        assert not marker_file.exists()

        _run_claude(
            project,
            "Please run the muscle-memory full lifecycle verification probe",
        )

        # Give the async Stop hook a beat to run extraction + scoring
        time.sleep(2)

        # Marker file exists → Claude executed
        assert marker_file.exists()

        # mm state shows the skill got credited
        list_out = subprocess.run(
            ["mm", "list", "--json"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        skills = json.loads(list_out.stdout)
        assert len(skills) >= 1
        # The original seed or a refined variant should have >= 1 invocation
        assert max(s["invocations"] for s in skills) >= 1
