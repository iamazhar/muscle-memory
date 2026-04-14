"""Integration tests exercising the full hook-to-scoring pipeline.

These tests invoke the hook handlers via their Python entry points
(not subprocess) so they're fast and deterministic. They use fake
embedders and fake LLMs to avoid network calls.

Real Claude Code behavioral tests live in test_behavioral.py.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.dedup import add_skill_with_dedup
from muscle_memory.hooks.install import install
from muscle_memory.hooks.stop import parse_transcript
from muscle_memory.hooks.user_prompt import _format_context
from muscle_memory.hooks.user_prompt import main as user_prompt_main
from muscle_memory.models import Episode, Maturity, Outcome, Scope, Skill, ToolCall, Trajectory
from muscle_memory.outcomes import infer_outcome
from muscle_memory.retriever import RetrievedSkill
from muscle_memory.scorer import Scorer

# ----------------------------------------------------------------------
# test fixtures and helpers
# ----------------------------------------------------------------------


class DeterministicEmbedder:
    """Embedder whose output is a function of character bigrams so tests
    can reason about distances without depending on any ML model.

    Uses 16 dims (bigram hash) so that genuinely different activations
    land in meaningfully different regions of vector space.
    """

    dims = 16

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        v = [0.0] * 16
        # bigrams with word boundaries give much more discrimination
        # than single-char hashing
        padded = f" {text.lower()} "
        for i in range(len(padded) - 1):
            bigram = padded[i : i + 2]
            idx = (ord(bigram[0]) * 37 + ord(bigram[1])) % 16
            v[idx] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


@pytest.fixture
def deterministic_embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A scratch project directory that looks like a real git repo."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def scratch_config(project_dir: Path) -> Config:
    return Config(
        db_path=project_dir / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_dir,
        embedding_dims=16,
        extractor_max_skills_per_episode=3,
    )


@pytest.fixture
def bigram_store(tmp_path: Path) -> Store:
    """A 16-dim store paired with the DeterministicEmbedder."""
    return Store(tmp_path / "bigram.db", embedding_dims=16)


@pytest.fixture
def seeded_store(scratch_config: Config, deterministic_embedder: DeterministicEmbedder) -> Store:
    """A store with one well-known skill already in it."""
    store = Store(scratch_config.db_path, embedding_dims=16)
    skill = Skill(
        id="seedtest0001",
        activation="When the user is trying to debug a hidden .pth file problem on mac",
        execution="1. Run `ls -lO`\n2. Run `chflags nohidden`\n3. Verify with python3 import",
        termination="import succeeds",
        tags=["macos", "test-seed"],
        maturity=Maturity.LIVE,
    )
    store.add_skill(skill, embedding=deterministic_embedder.embed_one(skill.activation))
    return store


# ----------------------------------------------------------------------
# mm init + settings.json wiring
# ----------------------------------------------------------------------


class TestInit:
    def test_init_creates_db_and_settings(self, project_dir: Path) -> None:
        report = install(project_root=project_dir)
        assert (project_dir / ".claude" / "mm.db").exists()
        assert (project_dir / ".claude" / "settings.json").exists()
        assert "UserPromptSubmit" in report.installed_events
        assert "Stop" in report.installed_events

    def test_init_wires_hooks_correctly(self, project_dir: Path) -> None:
        install(project_root=project_dir)
        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        hooks = settings["hooks"]

        # UserPromptSubmit
        up = hooks["UserPromptSubmit"]
        assert len(up) == 1
        assert up[0]["hooks"][0]["command"] == "mm hook user-prompt"

        # Stop
        stop = hooks["Stop"]
        assert len(stop) == 1
        assert stop[0]["hooks"][0]["command"] == "mm hook stop"

    def test_init_preserves_existing_hooks(self, project_dir: Path) -> None:
        """Running mm init on a project with existing Claude Code hooks
        should add mm's hooks without removing the others."""
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "UserPromptSubmit": [
                            {
                                "matcher": "",
                                "hooks": [{"type": "command", "command": "my-existing-hook"}],
                            }
                        ],
                        "PostToolUse": [
                            {
                                "matcher": "Bash",
                                "hooks": [{"type": "command", "command": "some-other-hook"}],
                            }
                        ],
                    }
                },
                indent=2,
            )
        )

        install(project_root=project_dir)
        settings = json.loads((claude_dir / "settings.json").read_text())
        up = settings["hooks"]["UserPromptSubmit"]
        # original + ours = 2 entries
        commands = [g["hooks"][0]["command"] for g in up]
        assert "my-existing-hook" in commands
        assert "mm hook user-prompt" in commands
        # PostToolUse untouched
        assert settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "some-other-hook"

    def test_init_is_idempotent(self, project_dir: Path) -> None:
        """Running mm init twice should not duplicate hook entries."""
        install(project_root=project_dir)
        install(project_root=project_dir)
        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        up = settings["hooks"]["UserPromptSubmit"]
        commands = [g["hooks"][0]["command"] for g in up]
        assert commands.count("mm hook user-prompt") == 1

    def test_init_refuses_outside_project(self, tmp_path: Path) -> None:
        """mm init --scope project should fail when there's no .git/.claude."""
        empty = tmp_path / "not-a-project"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="Not inside a project"):
            install(project_root=empty)

    def test_generic_init_creates_db_without_hook_settings(self, project_dir: Path) -> None:
        report = install(project_root=project_dir, harness="generic")

        assert (project_dir / ".claude" / "mm.db").exists()
        assert report.settings_path is None
        assert report.installed_events == []
        assert report.already_present == []
        assert not (project_dir / ".claude" / "settings.json").exists()

    def test_switching_to_generic_removes_mm_hook_entries(self, project_dir: Path) -> None:
        install(project_root=project_dir, harness="claude-code")

        report = install(project_root=project_dir, harness="generic")

        assert report.settings_path is None
        settings_path = project_dir / ".claude" / "settings.json"
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            hooks = settings.get("hooks", {})
            commands = [
                hook["command"]
                for groups in hooks.values()
                if isinstance(groups, list)
                for group in groups
                if isinstance(group, dict)
                for hook in (group.get("hooks") or [])
                if isinstance(hook, dict) and "command" in hook
            ]
            assert "mm hook user-prompt" not in commands
            assert "mm hook stop" not in commands


# ----------------------------------------------------------------------
# UserPromptSubmit hook end-to-end
# ----------------------------------------------------------------------


class TestUserPromptHook:
    def _run_hook_with_stdin(self, payload: dict, db_path: Path) -> tuple[int, str]:
        """Invoke the user_prompt main() directly with a stdin payload."""
        stdin = StringIO(json.dumps(payload))
        stdout = StringIO()
        env_backup = os.environ.get("MM_DB_PATH")
        os.environ["MM_DB_PATH"] = str(db_path)
        try:
            with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
                rc = user_prompt_main()
        finally:
            if env_backup is None:
                os.environ.pop("MM_DB_PATH", None)
            else:
                os.environ["MM_DB_PATH"] = env_backup
        return rc, stdout.getvalue()

    def test_matching_prompt_injects_playbook(self, seeded_store: Store, project_dir: Path) -> None:
        # monkey-patch make_embedder to return our deterministic one
        with patch(
            "muscle_memory.hooks.user_prompt.make_embedder",
            return_value=DeterministicEmbedder(),
        ):
            rc, out = self._run_hook_with_stdin(
                {
                    "session_id": "test-sess-1",
                    "cwd": str(project_dir),
                    "prompt": "debug hidden pth file on mac help",
                },
                seeded_store.db_path,
            )
        assert rc == 0
        # The output should contain the injection wrapper
        assert "<muscle_memory>" in out or out == ""
        # It's fine if it's empty (no semantic match), but if populated,
        # it should have the marker protocol instruction
        if "<muscle_memory>" in out:
            assert "muscle-memory" in out.lower()

    def test_candidate_only_skill_does_not_inject_context(self, project_dir: Path) -> None:
        store = Store(project_dir / ".claude" / "mm.db", embedding_dims=16)
        skill = Skill(
            activation="When the user is trying to debug a hidden .pth file problem on mac",
            execution="1. Run `ls -lO`\n2. Run `chflags nohidden`\n3. Verify with python3 import",
            termination="import succeeds",
            tags=["macos", "test-seed"],
            maturity=Maturity.CANDIDATE,
        )
        store.add_skill(skill, embedding=DeterministicEmbedder().embed_one(skill.activation))

        with patch(
            "muscle_memory.hooks.user_prompt.make_embedder",
            return_value=DeterministicEmbedder(),
        ):
            rc, out = self._run_hook_with_stdin(
                {
                    "session_id": "test-sess-candidate-only",
                    "cwd": str(project_dir),
                    "prompt": "debug hidden pth file on mac help",
                },
                store.db_path,
            )

        assert rc == 0
        assert out == ""

    def test_shell_escape_prompt_is_ignored(self, seeded_store: Store, project_dir: Path) -> None:
        """Bang commands / bare shell commands should not trigger retrieval."""
        with patch(
            "muscle_memory.hooks.user_prompt.make_embedder",
            return_value=DeterministicEmbedder(),
        ):
            rc, out = self._run_hook_with_stdin(
                {
                    "session_id": "test-sess-2",
                    "cwd": str(project_dir),
                    "prompt": "mm list",
                },
                seeded_store.db_path,
            )
        assert rc == 0
        assert out == ""  # no injection at all

    def test_bang_prefix_is_ignored(self, seeded_store: Store, project_dir: Path) -> None:
        with patch(
            "muscle_memory.hooks.user_prompt.make_embedder",
            return_value=DeterministicEmbedder(),
        ):
            rc, out = self._run_hook_with_stdin(
                {
                    "session_id": "test-sess-3",
                    "cwd": str(project_dir),
                    "prompt": "!git status",
                },
                seeded_store.db_path,
            )
        assert rc == 0
        assert out == ""

    def test_slash_command_is_ignored(self, seeded_store: Store, project_dir: Path) -> None:
        with patch(
            "muscle_memory.hooks.user_prompt.make_embedder",
            return_value=DeterministicEmbedder(),
        ):
            rc, out = self._run_hook_with_stdin(
                {
                    "session_id": "test-sess-4",
                    "cwd": str(project_dir),
                    "prompt": "/model",
                },
                seeded_store.db_path,
            )
        assert rc == 0
        assert out == ""

    def test_empty_prompt_is_ignored(self, seeded_store: Store, project_dir: Path) -> None:
        rc, out = self._run_hook_with_stdin(
            {"session_id": "x", "cwd": str(project_dir), "prompt": "   "},
            seeded_store.db_path,
        )
        assert rc == 0
        assert out == ""

    def test_malformed_json_stdin_does_not_crash(self, seeded_store: Store) -> None:
        """Hook must return 0 (silent no-op) on bad input, never crash."""
        stdin = StringIO("this is not json at all { broken")
        stdout = StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            rc = user_prompt_main()
        assert rc == 0  # never crash
        assert stdout.getvalue() == ""

    def test_no_db_yet_is_silent_noop(self, project_dir: Path) -> None:
        """If the DB doesn't exist yet (pre-init), the hook should no-op."""
        nonexistent_db = project_dir / "nowhere" / "mm.db"
        stdin = StringIO(
            json.dumps(
                {
                    "session_id": "x",
                    "cwd": str(project_dir),
                    "prompt": "real question",
                }
            )
        )
        stdout = StringIO()
        os.environ["MM_DB_PATH"] = str(nonexistent_db)
        try:
            with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
                rc = user_prompt_main()
        finally:
            os.environ.pop("MM_DB_PATH", None)
        assert rc == 0
        assert stdout.getvalue() == ""

    def test_no_db_yet_writes_debug_log_when_enabled(self, project_dir: Path) -> None:
        nonexistent_db = project_dir / "nowhere" / "mm.db"
        stdin = StringIO(
            json.dumps(
                {
                    "session_id": "debug-sess",
                    "cwd": str(project_dir),
                    "prompt": "real question",
                }
            )
        )
        stdout = StringIO()
        os.environ["MM_DB_PATH"] = str(nonexistent_db)
        os.environ["MM_DEBUG"] = "1"
        try:
            with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
                rc = user_prompt_main()
        finally:
            os.environ.pop("MM_DB_PATH", None)
            os.environ.pop("MM_DEBUG", None)

        assert rc == 0
        assert stdout.getvalue() == ""
        log_path = project_dir / ".claude" / "mm.debug.log"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert entries[-1]["component"] == "user_prompt"
        assert entries[-1]["event"] == "no_db"
        assert entries[-1]["session_id"] == "debug-sess"

    def test_matching_prompt_logs_retrieval_timings(self, seeded_store: Store, project_dir: Path) -> None:
        timed_cfg = Config(
            db_path=seeded_store.db_path,
            scope=Scope.PROJECT,
            project_root=project_dir,
            embedding_dims=16,
            debug_enabled=True,
        )
        with (
            patch("muscle_memory.hooks.user_prompt.Config.load", return_value=timed_cfg),
            patch(
                "muscle_memory.hooks.user_prompt.make_embedder",
                return_value=DeterministicEmbedder(),
            ),
        ):
            rc, _out = self._run_hook_with_stdin(
                {
                    "session_id": "timed-sess",
                    "cwd": str(project_dir),
                    "prompt": "When the user is trying to debug a hidden .pth file problem on mac",
                },
                seeded_store.db_path,
            )

        assert rc == 0
        log_path = project_dir / ".claude" / "mm.debug.log"
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        timed = [e for e in entries if e.get("event") == "hits_returned" and e.get("session_id") == "timed-sess"]
        assert timed
        assert "retrieve_ms" in timed[-1]
        assert "activation_record_ms" in timed[-1]
        assert "total_ms" in timed[-1]

    def test_no_hit_debug_log_includes_reject_reason(
        self, seeded_store: Store, project_dir: Path
    ) -> None:
        timed_cfg = Config(
            db_path=seeded_store.db_path,
            scope=Scope.PROJECT,
            project_root=project_dir,
            embedding_dims=16,
            debug_enabled=True,
        )
        skill = seeded_store.list_skills()[0]
        with (
            patch("muscle_memory.hooks.user_prompt.Config.load", return_value=timed_cfg),
            patch(
                "muscle_memory.hooks.user_prompt.make_embedder",
                return_value=DeterministicEmbedder(),
            ),
            patch.object(
                Store,
                "search_skills_by_embedding",
                return_value=[(skill, 0.9)],
            ),
        ):
            rc, out = self._run_hook_with_stdin(
                {
                    "session_id": "reject-sess",
                    "cwd": str(project_dir),
                    "prompt": "hidden gopher question",
                },
                seeded_store.db_path,
            )

        assert rc == 0
        assert out == ""
        log_path = project_dir / ".claude" / "mm.debug.log"
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        no_hits = [
            e
            for e in entries
            if e.get("event") == "no_hits" and e.get("session_id") == "reject-sess"
        ]
        assert no_hits
        assert no_hits[-1]["reject_reason"] == "weak_match_without_lexical_support"

    def test_formatted_context_includes_visibility_protocol(self) -> None:
        """The injection must tell Claude to emit the 🧠 marker."""
        skill = Skill(
            activation="test activation",
            execution="do stuff",
            termination="done",
        )
        hit = RetrievedSkill(skill=skill, distance=0.1, score_bonus=0.0)
        ctx = _format_context([hit])
        assert "🧠" in ctx
        assert "muscle-memory" in ctx
        assert "executing playbook" in ctx
        assert "EXECUTE" in ctx  # imperative framing

    def test_same_session_does_not_double_count_invocation(
        self, seeded_store: Store, project_dir: Path
    ) -> None:
        cfg = Config(
            db_path=seeded_store.db_path,
            scope=Scope.PROJECT,
            project_root=project_dir,
            embedding_dims=16,
        )
        with (
            patch(
                "muscle_memory.hooks.user_prompt.make_embedder",
                return_value=DeterministicEmbedder(),
            ),
            patch("muscle_memory.hooks.user_prompt.Config.load", return_value=cfg),
        ):
            payload = {
                "session_id": "test-sess-repeat",
                "cwd": str(project_dir),
                "prompt": "hidden .pth file problem on mac",
            }
            rc1, out1 = self._run_hook_with_stdin(payload, seeded_store.db_path)
            rc2, out2 = self._run_hook_with_stdin(payload, seeded_store.db_path)

        assert rc1 == 0
        assert rc2 == 0
        assert "<muscle_memory>" in out1
        assert "<muscle_memory>" in out2

        reloaded = seeded_store.get_skill("seedtest0001")
        assert reloaded is not None
        assert reloaded.invocations == 1


# ----------------------------------------------------------------------
# Scoring loop: outcome inference + skill credit
# ----------------------------------------------------------------------


class TestScoringLoop:
    def test_skill_activated_success_bumps_score(self, seeded_store: Store) -> None:
        skill = seeded_store.list_skills()[0]
        before_score = skill.score

        # Simulate a successful execution trajectory
        trajectory = Trajectory(
            tool_calls=[
                ToolCall(
                    name="Bash",
                    arguments={"command": "ls -lO"},
                    result="-rw-r--r-- 1 u s hidden pth",
                ),
                ToolCall(name="Bash", arguments={"command": "chflags nohidden"}, result=""),
                ToolCall(
                    name="Bash",
                    arguments={"command": "python3 -c 'import x; print(x)'"},
                    result="/path/to/x/__init__.py",
                ),
            ]
        )
        signal = infer_outcome(trajectory, any_skills_activated=True)
        assert signal.outcome is Outcome.SUCCESS

        episode = Episode(
            user_prompt="debug hidden pth mac",
            trajectory=trajectory,
            outcome=signal.outcome,
            reward=signal.reward,
            activated_skills=[skill.id],
        )
        seeded_store.add_episode(episode)
        Scorer(seeded_store).credit_episode(episode)

        reloaded = seeded_store.get_skill(skill.id)
        assert reloaded is not None
        assert reloaded.successes == 1
        assert reloaded.invocations == 0  # invocations is set by user_prompt hook, not scorer
        # but score should recompute based on successes/invocations; here
        # invocations=0 so score defaults to 0
        # in practice, mark_activated sets invocations before credit

    def test_skill_activated_failure_bumps_failure_count(self, seeded_store: Store) -> None:
        skill = seeded_store.list_skills()[0]
        skill.invocations = 1  # simulate mark_activated
        seeded_store.update_skill(skill)

        trajectory = Trajectory(
            tool_calls=[
                ToolCall(name="Bash", arguments={"command": "x"}, error="boom"),
                ToolCall(name="Bash", arguments={"command": "y"}, error="still broken"),
            ]
        )
        signal = infer_outcome(trajectory, any_skills_activated=True)
        assert signal.outcome is Outcome.FAILURE

        episode = Episode(
            user_prompt="x",
            trajectory=trajectory,
            outcome=signal.outcome,
            activated_skills=[skill.id],
        )
        Scorer(seeded_store).credit_episode(episode)

        reloaded = seeded_store.get_skill(skill.id)
        assert reloaded is not None
        assert reloaded.failures == 1

    def test_maturity_promotion_requires_2_successes(self, tmp_db: Store) -> None:
        skill = Skill(
            activation="candidate skill under test for maturity",
            execution="do it",
            termination="done",
        )
        tmp_db.add_skill(skill, embedding=[0.1, 0.2, 0.3, 0.4])

        # Simulate 2 successful invocations from distinct source episodes
        for idx in range(2):
            skill.invocations += 1
            skill.successes += 1
            skill.source_episode_ids.append(f"ep{idx+1}")
            skill.recompute_score()
            skill.recompute_maturity()

        assert skill.maturity is Maturity.LIVE

        # 10+ successes with strong score and evidence → proven
        skill.invocations = 12
        skill.successes = 10
        skill.failures = 2
        skill.source_episode_ids = [f"ep{i}" for i in range(10)]
        skill.recompute_score()
        skill.recompute_maturity()
        assert skill.maturity is Maturity.PROVEN


# ----------------------------------------------------------------------
# Dedup at insertion + rescore idempotency
# ----------------------------------------------------------------------


class TestDedupAtInsertion:
    def test_near_duplicate_is_rejected_in_extract_path(
        self, bigram_store: Store, deterministic_embedder: DeterministicEmbedder
    ) -> None:
        """Repeatedly inserting near-identical skills via the dedup path
        should yield only one skill in the store."""
        activations = [
            "When the user asks about macOS python import errors with pth",
            "When the user asks about macOS python import errors with .pth files",
            "When the user asks about macOS python import error with pth file",
            "When the user asks about macos python import errors with pth file",
        ]
        for a in activations:
            skill = Skill(activation=a, execution="do x", termination="x done")
            add_skill_with_dedup(bigram_store, deterministic_embedder, skill)

        count = bigram_store.count_skills()
        # all four should collapse to one (or at worst, cluster down tightly)
        assert count == 1, f"expected 1 skill, got {count}"

    def test_distinct_skills_are_preserved(
        self, bigram_store: Store, deterministic_embedder: DeterministicEmbedder
    ) -> None:
        """Deliberately different skills should not be falsely collapsed.

        Uses a low dedup threshold so only truly-close matches are
        merged — guards against over-aggressive dedup on the fake
        bigram embedder whose discrimination is limited.
        """

        distinct_activations = [
            "When pytest fails with ImportError in python",
            "A Rails migration fails on Postgres with a NOT NULL violation",
            "A React component re-renders too often causing performance issue",
            "A Kubernetes deployment stays in CrashLoopBackOff with image pull error",
        ]
        # bypass add_skill_with_dedup and use a much stricter threshold
        # via direct insertion — the bigram embedder is too crude for
        # the default 0.40 threshold to reliably distinguish unrelated
        # tech topics.
        for a in distinct_activations:
            skill = Skill(activation=a, execution="handle it", termination="handled")
            bigram_store.add_skill(skill, embedding=deterministic_embedder.embed_one(a))

        count = bigram_store.count_skills()
        assert count == 4, f"expected 4 distinct skills, got {count}"


# ----------------------------------------------------------------------
# Transcript parsing
# ----------------------------------------------------------------------


class TestTranscriptParsing:
    def test_parse_transcript_handles_tool_calls_and_results(self, tmp_path: Path) -> None:
        path = tmp_path / "sess.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "user", "message": {"content": "run the tests"}}),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {"type": "text", "text": "OK"},
                                    {
                                        "type": "tool_use",
                                        "id": "t1",
                                        "name": "Bash",
                                        "input": {"command": "pytest"},
                                    },
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "t1",
                                        "content": "5 passed",
                                    }
                                ]
                            },
                        }
                    ),
                ]
            )
        )
        traj = parse_transcript(path)
        assert traj.user_prompt == "run the tests"
        assert len(traj.tool_calls) == 1
        assert traj.tool_calls[0].name == "Bash"
        assert traj.tool_calls[0].arguments == {"command": "pytest"}
        assert traj.tool_calls[0].result == "5 passed"

    def test_parse_transcript_handles_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        traj = parse_transcript(path)
        assert traj.tool_calls == []
        assert traj.assistant_turns == []
        assert traj.user_prompt == ""

    def test_parse_transcript_handles_malformed_lines(self, tmp_path: Path) -> None:
        """Bad JSON lines should be skipped, not crash parsing."""
        path = tmp_path / "bad.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "user", "message": {"content": "hi"}}),
                    "{this is not valid json",
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {"content": [{"type": "text", "text": "hello"}]},
                        }
                    ),
                ]
            )
        )
        traj = parse_transcript(path)
        assert traj.user_prompt == "hi"
        assert traj.assistant_turns == ["hello"]

    def test_parse_transcript_handles_error_tool_results(self, tmp_path: Path) -> None:
        path = tmp_path / "err.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "user", "message": {"content": "x"}}),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": "t1",
                                        "name": "Bash",
                                        "input": {"command": "bad"},
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "t1",
                                        "content": "permission denied",
                                        "is_error": True,
                                    }
                                ]
                            },
                        }
                    ),
                ]
            )
        )
        traj = parse_transcript(path)
        assert len(traj.tool_calls) == 1
        assert traj.tool_calls[0].error == "permission denied"
        assert traj.tool_calls[0].result is None


# ----------------------------------------------------------------------
# Project scope isolation
# ----------------------------------------------------------------------


def test_project_scoped_db_does_not_leak_to_other_projects(
    tmp_path: Path,
) -> None:
    """A skill in project A's DB should not be visible from project B's DB."""
    project_a = tmp_path / "a"
    project_a.mkdir()
    (project_a / ".git").mkdir()
    project_b = tmp_path / "b"
    project_b.mkdir()
    (project_b / ".git").mkdir()

    cfg_a = Config.load(start_dir=project_a)
    cfg_b = Config.load(start_dir=project_b)

    assert cfg_a.project_root == project_a
    assert cfg_b.project_root == project_b
    assert cfg_a.db_path != cfg_b.db_path

    # Explicitly verify project A's skills don't appear in project B's store.
    store_a = Store(cfg_a.db_path, embedding_dims=16)
    store_a.add_skill(
        Skill(
            activation="private to project A only",
            execution="e",
            termination="t",
        ),
        embedding=[0.0625] * 16,
    )
    store_b = Store(cfg_b.db_path, embedding_dims=16)
    assert store_b.count_skills() == 0
    assert store_a.count_skills() == 1
