"""Claude Code `UserPromptSubmit` hook handler.

Reads a JSON payload on stdin, retrieves the most relevant Skills
for the user's current prompt, and prints an `<additional_context>`
block to stdout (which Claude Code appends to the prompt).

Also records the activated skill IDs in a per-session sidecar file
so the matching `Stop` hook can credit them later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.embeddings import make_embedder
from muscle_memory.retriever import RetrievedSkill, Retriever


def main(argv: list[str] | None = None) -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        # bad input — don't block the user
        return 0

    prompt = _extract_prompt(payload)
    if not prompt:
        return 0

    cwd = payload.get("cwd")
    session_id = payload.get("session_id") or ""

    try:
        cfg = Config.load(start_dir=Path(cwd) if cwd else None)
        # if there's no DB yet, silently bail
        if not cfg.db_path.exists():
            return 0

        store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
        embedder = make_embedder(cfg)
        retriever = Retriever(store, embedder, cfg)

        hits = retriever.retrieve(prompt)
    except Exception:
        # hook failures must never break the user's turn
        return 0

    if not hits:
        return 0

    try:
        retriever.mark_activated(hits)
        _record_activation(cfg, session_id, [h.skill.id for h in hits])
    except Exception:
        pass  # nice-to-have, not critical

    context = _format_context(hits)
    sys.stdout.write(context)
    sys.stdout.flush()
    return 0


def _extract_prompt(payload: dict[str, Any]) -> str:
    for key in ("prompt", "user_prompt", "message", "text"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    # nested: payload['hook_event'] may contain the prompt
    hook_data = payload.get("hook_event") or {}
    if isinstance(hook_data, dict):
        for key in ("prompt", "user_prompt", "message"):
            if isinstance(hook_data.get(key), str):
                return hook_data[key]
    return ""


def _format_context(hits: list[RetrievedSkill]) -> str:
    lines = [
        "<muscle_memory>",
        "These procedural skills were learned from past sessions in this project.",
        "Treat them as strong hints, not hard rules. Ignore any that don't fit the task.",
        "",
    ]
    for i, hit in enumerate(hits, start=1):
        s = hit.skill
        lines.append(f"## Skill {i} — {s.maturity.value} (score {s.score:.2f})")
        lines.append(f"**When:** {s.activation}")
        lines.append("**Do:**")
        lines.append(s.execution)
        lines.append(f"**Stop when:** {s.termination}")
        if s.tool_hints:
            lines.append(f"**Tool hints:** {', '.join(s.tool_hints)}")
        lines.append("")
    lines.append("</muscle_memory>")
    return "\n".join(lines)


def _record_activation(
    cfg: Config, session_id: str, skill_ids: list[str]
) -> None:
    if not session_id:
        return
    activations_dir = cfg.db_path.parent / "mm.activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    sidecar = activations_dir / f"{session_id}.json"

    existing: list[str] = []
    if sidecar.exists():
        try:
            existing = json.loads(sidecar.read_text())
        except Exception:
            existing = []
    merged = list(dict.fromkeys([*existing, *skill_ids]))
    sidecar.write_text(json.dumps(merged))


if __name__ == "__main__":
    sys.exit(main())
