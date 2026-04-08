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
from muscle_memory.models import Episode, ToolCall, Trajectory
from muscle_memory.outcomes import infer_outcome
from muscle_memory.scorer import Scorer


def main(argv: list[str] | None = None) -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    session_id = payload.get("session_id") or ""
    transcript_path_str = payload.get("transcript_path") or ""
    cwd = payload.get("cwd")

    if not transcript_path_str:
        return 0
    transcript_path = Path(transcript_path_str).expanduser()
    if not transcript_path.exists():
        return 0

    try:
        cfg = Config.load(start_dir=Path(cwd) if cwd else None)
        if not cfg.db_path.exists():
            return 0

        trajectory = parse_transcript(transcript_path)
        if not trajectory.tool_calls and not trajectory.assistant_turns:
            return 0

        signal = infer_outcome(trajectory)

        activated = _load_activations(cfg, session_id)

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
        Scorer(store, max_skills=cfg.max_skills).credit_episode(episode)

        # Fire async extraction in a detached subprocess. We do not
        # wait — Claude Code's hook has to return quickly.
        _fire_async_extraction(episode.id, cfg.db_path)

    except Exception:
        # never break the user's shutdown path
        return 0

    return 0


def parse_transcript(path: Path) -> Trajectory:
    """Parse a Claude Code session JSONL into a Trajectory."""
    user_prompt = ""
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
                            elif block.get("type") == "text" and not user_prompt:
                                user_prompt = block.get("text", "")

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
    if not session_id:
        return []
    sidecar = cfg.db_path.parent / "mm.activations" / f"{session_id}.json"
    if not sidecar.exists():
        return []
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return []


def _fire_async_extraction(episode_id: str, db_path: Path) -> None:
    """Start a detached `mm extract-episode <id>` process.

    Uses `start_new_session=True` and redirects stdio so the subprocess
    fully detaches from the hook.
    """
    try:
        subprocess.Popen(  # noqa: S603 - user-trusted command
            [sys.executable, "-m", "muscle_memory", "extract-episode", episode_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "MM_DB_PATH": str(db_path)},
            close_fds=True,
        )
    except Exception:
        # if async fails, write to a queue file for later processing
        queue_path = db_path.parent / "mm.extract_queue.txt"
        try:
            with queue_path.open("a", encoding="utf-8") as f:
                f.write(f"{episode_id}\n")
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
