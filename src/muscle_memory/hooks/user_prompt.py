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


def _skill_title(activation: str, max_len: int = 60) -> str:
    """Derive a short human-readable title from the activation text.

    Used for the visible acknowledgment line so the user can see which
    playbook fired. Trimmed to fit in one line of terminal output.
    """
    s = activation.strip()
    # strip leading "When "
    if s.lower().startswith("when "):
        s = s[5:]
    # cut at first period or comma
    for stop in (". ", ", "):
        i = s.find(stop)
        if 10 < i < max_len:
            s = s[:i]
            break
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def _format_context(hits: list[RetrievedSkill]) -> str:
    """Format retrieved skills as imperative playbooks to execute.

    Two principles:
      1. Imperative framing — "EXECUTE the steps", not "treat as hints".
         Softening language gets the skill narrated back to the user
         instead of actually run.
      2. Visible acknowledgment — the wrapper tells Claude to prefix
         its response with a one-line marker naming the playbook, so
         the user can see in real time that muscle-memory is working.
    """
    titles = [_skill_title(h.skill.activation) for h in hits]
    titles_list = " | ".join(f'"{t}"' for t in titles)

    lines = [
        "<muscle_memory>",
        "These are verified playbooks extracted from past successful sessions",
        "in this project. For each playbook below, if the `Activate when`",
        "condition clearly matches the user's current situation, **EXECUTE the",
        "Steps directly**: run the commands, make the edits, verify the result.",
        "Do not just describe the steps to the user — actually perform them.",
        "The user wants the problem fixed, not a list of instructions.",
        "",
        "If a playbook's `Activate when` clearly does NOT fit the current task,",
        "ignore it and proceed normally.",
        "",
        "### Visibility protocol (required)",
        "",
        "Begin your response with ONE line in exactly this format so the user",
        "can see which playbook fired:",
        "",
        f"> 🧠 **muscle-memory**: executing playbook — <title>",
        "",
        f"Where `<title>` is one of: {titles_list}",
        "",
        "If NONE of the playbooks apply to the current task, instead start",
        "with: `🧠 **muscle-memory**: no matching playbook, proceeding normally`",
        "Then continue with the actual work. Do not explain muscle-memory,",
        "do not discuss the playbook metadata — acknowledge, then act.",
        "",
    ]
    for i, (hit, title) in enumerate(zip(hits, titles), start=1):
        s = hit.skill
        lines.append(
            f"## Playbook {i} — \"{title}\""
            f" · {s.maturity.value}"
            f" · {s.successes}/{s.invocations} successes"
        )
        lines.append(f"**Activate when:** {s.activation}")
        lines.append("**Steps (execute in order):**")
        lines.append(s.execution)
        lines.append(f"**Done when:** {s.termination}")
        if s.tool_hints:
            lines.append(f"**Preferred tools:** {', '.join(s.tool_hints)}")
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
