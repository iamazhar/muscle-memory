"""Claude Code `Stop` hook handler.

When a session ends, parse the transcript, build an Episode, infer
the outcome, credit the skills that were active during the session,
and fire off async extraction of new skills from the trajectory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.debug import log_debug_event
from muscle_memory.models import BackgroundJob, Episode, JobKind, JobStatus, ToolCall, Trajectory
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

    try:
        session_id = payload.get("session_id") or ""
        transcript_path_str = payload.get("transcript_path") or ""
        cwd = payload.get("cwd")
    except Exception:
        return 0

    cfg: Config | None = None
    try:
        cfg = Config.load(start_dir=Path(cwd) if cwd else None)
        if cfg.project_root is None and cwd:
            cfg.project_root = Path(cwd)
    except Exception:
        cfg = None

    if not isinstance(transcript_path_str, str):
        return 0

    if not transcript_path_str:
        if cfg is not None:
            log_debug_event(cfg, component="stop", event="missing_transcript_path", session_id=session_id)
        return 0
    transcript_path = Path(transcript_path_str).expanduser()
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
        if not cfg.db_path.exists():
            log_debug_event(cfg, component="stop", event="no_db", session_id=session_id)
            return 0
        if (cfg.db_path.parent / "mm.paused").exists():
            log_debug_event(cfg, component="stop", event="paused", session_id=session_id)
            return 0

        trajectory = parse_transcript(transcript_path)
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
        # some skill meets the auto-refine criteria (failures ≥ 2,
        # score ≤ 0.6, invocations ≥ 5). Runs detached.
        if _any_skill_needs_refinement(store):
            refine_job = BackgroundJob(kind=JobKind.REFINE, payload={})
            store.add_job(refine_job)
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


def parse_transcript(path: Path) -> Trajectory:
    """Parse a Claude Code session JSONL into a Trajectory."""
    user_prompt = ""
    user_followups: list[str] = []
    tool_calls: list[ToolCall] = []
    assistant_turns: list[str] = []
    # index tool uses by id so we can attach results
    pending_by_id: dict[str, ToolCall] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec_type = rec.get("type")
            msg = rec.get("message") or {}

            if rec_type == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    if not user_prompt:
                        user_prompt = content
                    else:
                        user_followups.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                tool_id = block.get("tool_use_id")
                                result_content = block.get("content")
                                is_error = block.get("is_error", False)
                                result_text = _flatten_content(result_content)
                                if tool_id and tool_id in pending_by_id:
                                    tc = pending_by_id[tool_id]
                                    if is_error:
                                        tc.error = result_text
                                    else:
                                        tc.result = result_text
                            elif block.get("type") == "text":
                                text = block.get("text", "")
                                if not user_prompt:
                                    user_prompt = text
                                else:
                                    user_followups.append(text)

            elif rec_type == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            assistant_turns.append(block.get("text", ""))
                        elif btype == "tool_use":
                            tc = ToolCall(
                                name=block.get("name", "unknown"),
                                arguments=block.get("input", {}) or {},
                            )
                            tool_calls.append(tc)
                            tool_id = block.get("id")
                            if tool_id:
                                pending_by_id[tool_id] = tc

    return Trajectory(
        user_prompt=user_prompt,
        user_followup=" ".join(user_followups),
        tool_calls=tool_calls,
        assistant_turns=assistant_turns,
    )


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
        if store is not None:
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
        if store is not None:
            store.update_job_status(job_id, status=JobStatus.FAILED, error=str(exc))
        pass  # best-effort; never block the hook


if __name__ == "__main__":
    sys.exit(main())
