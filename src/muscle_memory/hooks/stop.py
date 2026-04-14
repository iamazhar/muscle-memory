"""Claude Code `Stop` hook handler.

When a session ends, parse the transcript, build an Episode, infer
the outcome, credit the skills that were active during the session,
and fire off async extraction of new skills from the trajectory.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.debug import log_debug_event
from muscle_memory.harness import get_harness
from muscle_memory.models import BackgroundJob, Episode, JobKind, JobStatus, Trajectory
from muscle_memory.outcomes import infer_outcome
from muscle_memory.scorer import Scorer


def main(argv: list[str] | None = None) -> int:
    # Hook failures MUST NEVER crash the user's Claude Code session.
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    if not isinstance(payload, dict):
        return 0
    cfg: Config | None = None
    session_id = ""
    cwd: str | None = None
    transcript_path: Path | None = None

    try:
        raw_cwd = payload.get("cwd")
        cwd_hint = Path(raw_cwd) if isinstance(raw_cwd, str) and raw_cwd else None
        cfg = Config.load(start_dir=cwd_hint)
        if cfg.project_root is None and cwd_hint is not None:
            cfg.project_root = cwd_hint
        adapter = get_harness(cfg.harness)
        session_id = adapter.extract_session_id(payload)
        cwd_path = adapter.extract_cwd(payload)
        cwd = str(cwd_path) if cwd_path is not None else (raw_cwd if isinstance(raw_cwd, str) else None)
        transcript_path = adapter.extract_transcript_path(payload)
    except Exception:
        return 0

    if transcript_path is None:
        if cfg is not None:
            log_debug_event(cfg, component="stop", event="missing_transcript_path", session_id=session_id)
        return 0

    if not transcript_path.exists():
        if cfg is not None:
            log_debug_event(
                cfg,
                component="stop",
                event="missing_transcript",
                session_id=session_id,
                transcript_path=str(transcript_path),
            )
        return 0

    try:
        if cfg is None:
            cfg = Config.load(start_dir=Path(cwd) if cwd else None)
        adapter = get_harness(cfg.harness)
        if not cfg.db_path.exists():
            log_debug_event(cfg, component="stop", event="no_db", session_id=session_id)
            return 0
        if (cfg.db_path.parent / "mm.paused").exists():
            log_debug_event(cfg, component="stop", event="paused", session_id=session_id)
            return 0

        trajectory = adapter.parse_transcript(transcript_path)
        if not trajectory.tool_calls and not trajectory.assistant_turns:
            log_debug_event(cfg, component="stop", event="empty_trajectory", session_id=session_id)
            return 0

        activated = _load_activations(cfg, session_id)
        signal = infer_outcome(
            trajectory,
            user_followup=trajectory.user_followup,
            any_skills_activated=bool(activated),
        )

        episode = Episode(
            session_id=session_id,
            user_prompt=trajectory.user_prompt or "(unknown)",
            trajectory=trajectory,
            outcome=signal.outcome,
            reward=signal.reward,
            project_path=cwd,
            activated_skills=activated,
        )

        store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
        store.add_episode(episode)
        updated_skills = Scorer(store, max_skills=cfg.max_skills).credit_episode(episode)
        log_debug_event(
            cfg,
            component="stop",
            event="episode_recorded",
            session_id=session_id,
            episode_id=episode.id,
            outcome=signal.outcome.value,
            reward=signal.reward,
            activated_skills=activated,
            credited_skill_ids=[skill.id for skill in updated_skills],
            tool_call_count=len(trajectory.tool_calls),
        )

        extraction_job = BackgroundJob(kind=JobKind.EXTRACT, payload={"episode_id": episode.id})
        store.add_job(extraction_job)

        # Fire async extraction in a detached subprocess. We do not
        # wait — Claude Code's hook has to return quickly.
        _fire_async_extraction(episode.id, cfg.db_path, job_id=extraction_job.id)
        log_debug_event(
            cfg,
            component="stop",
            event="spawned_extraction",
            session_id=session_id,
            episode_id=episode.id,
            job_id=extraction_job.id,
        )

        # Notify via Claude Code's hook response JSON
        response = {"stopReason": "🧠 muscle-memory: extracting skills from this session…"}
        sys.stdout.write(json.dumps(response))
        sys.stdout.flush()

        # v0.2: also fire a refinement sweep. It's a no-op unless
        # auto-refinement is enabled, some skill meets the auto-refine
        # criteria (failures ≥ 2, score ≤ 0.6, invocations ≥ 5), and
        # no refine job is already pending or running.
        if cfg.auto_refine_enabled and _any_skill_needs_refinement(store) and not _has_running_job(
            store, JobKind.REFINE
        ):
            refine_job = BackgroundJob(kind=JobKind.REFINE, payload={})
            try:
                store.add_job(refine_job)
            except sqlite3.IntegrityError:
                refine_job = None
            if refine_job is None:
                return 0
            _fire_async_refinement(cfg.db_path, job_id=refine_job.id)
            log_debug_event(
                cfg,
                component="stop",
                event="spawned_refinement",
                session_id=session_id,
                episode_id=episode.id,
                job_id=refine_job.id,
            )

    except Exception as exc:
        # never break the user's shutdown path
        if cfg is not None:
            log_debug_event(cfg, component="stop", event="hook_error", session_id=session_id, error=str(exc))
        return 0

    return 0


def _any_skill_needs_refinement(store: Store) -> bool:
    """Cheap pre-check: does any skill meet the auto-refine criteria?

    Avoids spawning a subprocess when refinement would be a no-op.
    """
    try:
        from muscle_memory.refine import should_auto_refine

        for skill in store.list_skills():
            if should_auto_refine(skill):
                return True
    except Exception:
        pass
    return False


def _has_running_job(store: Store, kind: JobKind) -> bool:
    """Return True when a job of `kind` is already pending or running."""
    try:
        for job in store.list_jobs(limit=None, kind=kind):
            if job.status in {JobStatus.PENDING, JobStatus.RUNNING}:
                return True
    except Exception:
        pass
    return False


def parse_transcript(path: Path) -> Trajectory:
    """Backward-compatible wrapper around the Claude Code harness parser."""
    adapter = get_harness("claude-code")
    return adapter.parse_transcript(path)


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _load_activations(cfg: Config, session_id: str) -> list[str]:
    """Load activated skill IDs from sidecar file.

    Handles both old format (list of strings) and new format
    (list of dicts with skill_id + distance).
    """
    if not session_id:
        return []
    sidecar = cfg.db_path.parent / "mm.activations" / f"{session_id}.json"
    if not sidecar.exists():
        return []
    try:
        raw = json.loads(sidecar.read_text())
        result: list[str] = []
        for entry in raw:
            if isinstance(entry, str):
                result.append(entry)
            elif isinstance(entry, dict) and "skill_id" in entry:
                result.append(entry["skill_id"])
        return result
    except Exception:
        return []


def _fire_async_extraction(episode_id: str, db_path: Path, *, job_id: str | None = None) -> None:
    """Start a detached `mm extract-episode <id>` process.

    Uses `start_new_session=True` and redirects stdio so the subprocess
    fully detaches from the hook.
    """
    store: Store | None = None
    if job_id is not None:
        store = Store(db_path)
        store.update_job_status(job_id, status=JobStatus.RUNNING)
    try:
        cmd = [sys.executable, "-m", "muscle_memory", "extract-episode", episode_id]
        if job_id is not None:
            cmd.extend(["--job-id", job_id])
        subprocess.Popen(  # noqa: S603 - user-trusted command
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "MM_DB_PATH": str(db_path)},
            close_fds=True,
        )
    except Exception as exc:
        if store is not None and job_id is not None:
            store.update_job_status(job_id, status=JobStatus.FAILED, error=str(exc))
        # if async fails, write to a queue file for later processing
        queue_path = db_path.parent / "mm.extract_queue.txt"
        try:
            with queue_path.open("a", encoding="utf-8") as f:
                f.write(f"{episode_id}\n")
        except Exception:
            pass


def _fire_async_refinement(db_path: Path, *, job_id: str | None = None) -> None:
    """Start a detached `mm refine --auto` sweep in the background.

    Runs after the extractor so it sees the latest skill state. This
    is the v0.2 integration point — the Scorer may have marked some
    skills eligible for auto-refinement after the current episode's
    credit pass, and we kick off a background refinement sweep here.
    """
    store: Store | None = None
    if job_id is not None:
        store = Store(db_path)
        store.update_job_status(job_id, status=JobStatus.RUNNING)
    try:
        cmd = [sys.executable, "-m", "muscle_memory", "refine", "--auto"]
        if job_id is not None:
            cmd.extend(["--job-id", job_id])
        subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "MM_DB_PATH": str(db_path)},
            close_fds=True,
        )
    except Exception as exc:
        if store is not None and job_id is not None:
            store.update_job_status(job_id, status=JobStatus.FAILED, error=str(exc))
        pass  # best-effort; never block the hook


if __name__ == "__main__":
    sys.exit(main())
