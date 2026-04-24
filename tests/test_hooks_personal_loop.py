"""Tests for hook integration with task and activation records."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.hooks import stop, user_prompt
from muscle_memory.models import DeliveryMode, Maturity, Outcome, Scope, Skill


class DummyEmbedder:
    dims = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        tokens = text.lower()
        return [
            1.0 if "pytest" in tokens else 0.0,
            1.0 if "import" in tokens else 0.0,
            1.0 if "runner" in tokens else 0.0,
            0.0,
        ]


def _config(project_root: Path) -> Config:
    return Config(
        db_path=project_root / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=project_root,
        embedding_dims=4,
        harness="claude-code",
        auto_refine_enabled=False,
    )


def _seed_skill(store: Store, embedder: DummyEmbedder) -> Skill:
    skill = Skill(
        activation="When pytest fails with import errors",
        execution="1. Use the repo test runner\n2. Rerun pytest",
        termination="Tests pass",
        maturity=Maturity.LIVE,
        successes=2,
        invocations=2,
        score=1.0,
        source_episode_ids=["ep1", "ep2"],
    )
    store.add_skill(skill, embedding=embedder.embed_one(skill.activation))
    return skill


def test_user_prompt_hook_records_task_and_activation(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "prompt": "pytest import errors",
    }

    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.user_prompt.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.user_prompt.make_embedder", return_value=embedder),
    ):
        exit_code = user_prompt.main([])

    assert exit_code == 0
    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 1
    assert tasks[0].session_id == "session-1"
    activations = store.list_activations_for_task(tasks[0].id)
    assert len(activations) == 1
    assert activations[0].skill_id == skill.id
    assert activations[0].delivery_mode is DeliveryMode.CLAUDE_HOOK


def test_repeated_user_prompt_hook_keeps_token_evidence_and_dedupes_credit(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)
    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "prompt": "pytest import errors",
    }

    with (
        patch("muscle_memory.hooks.user_prompt.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.user_prompt.make_embedder", return_value=embedder),
    ):
        monkeypatch.setattr("sys.stdin", _stdin(payload))
        assert user_prompt.main([]) == 0
        monkeypatch.setattr("sys.stdin", _stdin(payload))
        assert user_prompt.main([]) == 0

    tasks = store.list_tasks(limit=10)
    activations = [
        activation
        for task in tasks
        for activation in store.list_activations_for_task(task.id)
    ]
    assert len(tasks) == 2
    assert len(activations) == 2
    assert {activation.skill_id for activation in activations} == {skill.id}
    latest_task = tasks[0]
    latest_activations = store.list_activations_for_task(latest_task.id)
    assert len(latest_activations) == 1
    assert latest_activations[0].injected_token_count > 0
    updated_skill = store.get_skill(skill.id)
    assert updated_skill is not None
    assert updated_skill.invocations == 3
    assert updated_skill.successes == 2

    transcript = _write_success_transcript(project_root)
    payload["transcript_path"] = str(transcript)
    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.stop._fire_async_extraction"),
    ):
        assert stop.main([]) == 0

    credited_skill = store.get_skill(skill.id)
    assert credited_skill is not None
    assert credited_skill.invocations == 3
    assert credited_skill.successes == 3
    assert credited_skill.successes <= credited_skill.invocations
    measurement = store.get_measurement_for_task(latest_task.id)
    assert measurement is not None
    assert measurement.injected_skill_tokens > 0


def test_stop_hook_credits_canonical_activation_and_records_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    from muscle_memory.personal_loop import capture_task, record_activations
    from muscle_memory.retriever import RetrievedSkill

    task = capture_task(
        store,
        raw_prompt="pytest import errors",
        cleaned_prompt="pytest import errors",
        harness="claude-code",
        project_path=str(project_root),
        session_id="session-1",
    )
    record_activations(
        store,
        task=task,
        hits=[RetrievedSkill(skill=skill, distance=0.2, score_bonus=0.1)],
        delivery_mode=DeliveryMode.CLAUDE_HOOK,
        context_token_count=40,
    )

    transcript = project_root / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {"type": "user", "message": {"content": "pytest import errors"}}
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": "pytest"},
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
                                    "content": "5 passed in 1.0s",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "transcript_path": str(transcript),
    }

    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.stop._fire_async_extraction"),
    ):
        exit_code = stop.main([])

    assert exit_code == 0
    credited = store.list_activations_for_task(task.id)[0]
    assert credited.credited_outcome is Outcome.SUCCESS
    measurement = store.get_measurement_for_task(task.id)
    assert measurement is not None
    assert measurement.outcome is Outcome.SUCCESS
    assert measurement.injected_skill_tokens == 40


def test_stop_hook_still_spawns_extraction_when_measurement_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    from muscle_memory.personal_loop import capture_task, record_activations
    from muscle_memory.retriever import RetrievedSkill

    task = capture_task(
        store,
        raw_prompt="pytest import errors",
        cleaned_prompt="pytest import errors",
        harness="claude-code",
        project_path=str(project_root),
        session_id="session-1",
    )
    record_activations(
        store,
        task=task,
        hits=[RetrievedSkill(skill=skill, distance=0.2, score_bonus=0.1)],
        delivery_mode=DeliveryMode.CLAUDE_HOOK,
        context_token_count=40,
    )
    transcript = _write_success_transcript(project_root)
    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "transcript_path": str(transcript),
    }

    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch(
            "muscle_memory.hooks.stop.add_measurement_for_task",
            side_effect=RuntimeError("measurement failed"),
        ),
        patch("muscle_memory.hooks.stop._fire_async_extraction") as fire_async,
    ):
        exit_code = stop.main([])

    assert exit_code == 0
    assert fire_async.called
    jobs = store.list_jobs(limit=None)
    assert len(jobs) == 1


def test_stop_hook_falls_back_to_sidecar_when_task_has_no_canonical_activations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path
    (project_root / ".claude").mkdir()
    cfg = _config(project_root)
    store = Store(cfg.db_path, embedding_dims=4)
    embedder = DummyEmbedder()
    skill = _seed_skill(store, embedder)

    from muscle_memory.personal_loop import capture_task

    capture_task(
        store,
        raw_prompt="pytest import errors",
        cleaned_prompt="pytest import errors",
        harness="claude-code",
        project_path=str(project_root),
        session_id="session-1",
    )
    sidecar_dir = cfg.db_path.parent / "mm.activations"
    sidecar_dir.mkdir()
    (sidecar_dir / "session-1.json").write_text(
        json.dumps([{"skill_id": skill.id, "distance": 0.2}]),
        encoding="utf-8",
    )
    transcript = _write_success_transcript(project_root)
    payload = {
        "session_id": "session-1",
        "cwd": str(project_root),
        "transcript_path": str(transcript),
    }

    monkeypatch.setattr("sys.stdin", _stdin(payload))
    with (
        patch("muscle_memory.hooks.stop.Config.load", return_value=cfg),
        patch("muscle_memory.hooks.stop._fire_async_extraction") as fire_async,
    ):
        exit_code = stop.main([])

    assert exit_code == 0
    assert fire_async.called
    updated_skill = store.get_skill(skill.id)
    assert updated_skill is not None
    assert updated_skill.invocations == 2
    assert updated_skill.successes == 3
    task = store.find_latest_task_by_session("session-1")
    assert task is not None
    assert store.get_measurement_for_task(task.id) is None


def _write_success_transcript(project_root: Path) -> Path:
    transcript = project_root / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {"type": "user", "message": {"content": "pytest import errors"}}
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": "pytest"},
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
                                    "content": "5 passed in 1.0s",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    return transcript


def _stdin(payload: dict):
    class Stdin:
        def read(self) -> str:
            return json.dumps(payload)

    return Stdin()
