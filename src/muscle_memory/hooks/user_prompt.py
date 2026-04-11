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
from typing import Any, TypedDict

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.embeddings import make_embedder
from muscle_memory.retriever import RetrievedSkill, Retriever


class ActivationRecord(TypedDict):
    skill_id: str
    distance: float | None


def main(argv: list[str] | None = None) -> int:
    # Hook failures MUST NEVER crash the user's Claude Code session.
    # We aggressively catch everything — bad stdin, bad payload shape,
    # config errors, store errors, the works.
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    if not isinstance(payload, dict):
        return 0

    try:
        prompt = _extract_prompt(payload)
    except Exception:
        return 0

    if not prompt:
        return 0

    # Skip direct shell-escape commands: the user is running a tool,
    # not asking a question. Matching skills against "mm list" or "ls"
    # just bloats invocation counts without signal.
    if _is_shell_escape(prompt):
        return 0

    cwd = payload.get("cwd")
    session_id = payload.get("session_id") or ""

    try:
        cfg = Config.load(start_dir=Path(cwd) if cwd else None)
        # if there's no DB yet, silently bail
        if not cfg.db_path.exists():
            return 0
        # if paused, silently bail
        if (cfg.db_path.parent / "mm.paused").exists():
            return 0

        store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
        embedder = make_embedder(cfg)
        retriever = Retriever(store, embedder, cfg)

        hits = retriever.retrieve(prompt)
        existing_activation_ids = _load_recorded_activation_ids(cfg, session_id)
    except Exception:
        # hook failures must never break the user's turn
        return 0

    if not hits:
        return 0

    try:
        new_hits = [h for h in hits if h.skill.id not in existing_activation_ids]
        if new_hits:
            retriever.mark_activated(new_hits)
        _record_activation(
            cfg,
            session_id,
            [{"skill_id": h.skill.id, "distance": h.distance} for h in hits],
        )
    except Exception:
        pass  # nice-to-have, not critical

    context = _format_context(hits)
    sys.stdout.write(context)
    sys.stdout.flush()
    return 0


def _extract_prompt(payload: dict[str, Any]) -> str:
    for key in ("prompt", "user_prompt", "message", "text"):
        val = payload.get(key)
        if isinstance(val, str):
            return val
    # nested: payload['hook_event'] may contain the prompt
    hook_data = payload.get("hook_event") or {}
    if isinstance(hook_data, dict):
        for key in ("prompt", "user_prompt", "message"):
            nested_val = hook_data.get(key)
            if isinstance(nested_val, str):
                return nested_val
    return ""


# Prefixes that indicate the user is running a direct command rather
# than asking a natural-language question. We skip skill retrieval for
# these — matching "mm list" or "ls src/" against activation conditions
# just bloats invocation counts without adding signal.
_BANG_PREFIXES = (
    "!",
    "/",  # slash commands like /model, /clear
)


# Common shell command names that strongly suggest direct execution
# (when the user prompt is just `foo bar` with no natural-language shape).
_SHELL_CMD_HEADS = {
    "ls",
    "cd",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "pwd",
    "git",
    "npm",
    "yarn",
    "uv",
    "pip",
    "python",
    "python3",
    "node",
    "bash",
    "sh",
    "zsh",
    "which",
    "whoami",
    "env",
    "export",
    "echo",
    "mm",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "touch",
    "chmod",
    "chflags",
    "curl",
    "wget",
    "ssh",
    "scp",
    "kubectl",
    "docker",
}


def _is_shell_escape(prompt: str) -> bool:
    """Heuristic: does this prompt look like a direct shell command?

    Returns True for:
      - Anything starting with `!` (Claude Code shell-escape)
      - Anything starting with `/` (Claude Code slash command)
      - A single short line whose first word is a common shell command
    """
    s = prompt.strip()
    if not s:
        return True
    if s.startswith(_BANG_PREFIXES):
        return True
    # single-line, short, first token is a known command → probably direct
    if "\n" not in s and len(s) < 120:
        first_token = s.split(None, 1)[0].lower()
        if first_token in _SHELL_CMD_HEADS:
            return True
    return False


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
        "> 🧠 **muscle-memory**: executing playbook — <title>",
        "",
        f"Where `<title>` is one of: {titles_list}",
        "",
        "If NONE of the playbooks apply to the current task, do NOT emit any",
        "muscle-memory marker. Just proceed normally with the user's request.",
        "Do not explain muscle-memory or discuss the playbook metadata.",
        "",
    ]
    for i, (hit, title) in enumerate(zip(hits, titles), start=1):
        s = hit.skill
        lines.append(
            f'## Playbook {i} — "{title}"'
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
    cfg: Config,
    session_id: str,
    activations: list[ActivationRecord],
) -> None:
    """Record which skills were activated with their retrieval distances.

    Format: [{"skill_id": "abc", "distance": 0.42}, ...]
    Backward compatible: _load_activations in stop.py handles both
    old format (list of strings) and new format (list of dicts).
    """
    if not session_id:
        return
    activations_dir = cfg.db_path.parent / "mm.activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    sidecar = activations_dir / f"{session_id}.json"

    existing: list[ActivationRecord] = []
    if sidecar.exists():
        try:
            raw = json.loads(sidecar.read_text())
            # Handle old format: convert strings to dicts
            for entry in raw:
                if isinstance(entry, str):
                    existing.append({"skill_id": entry, "distance": None})
                elif isinstance(entry, dict):
                    skill_id = entry.get("skill_id")
                    distance = entry.get("distance")
                    if isinstance(skill_id, str) and (
                        distance is None or isinstance(distance, int | float)
                    ):
                        existing.append(
                            {
                                "skill_id": skill_id,
                                "distance": float(distance) if distance is not None else None,
                            }
                        )
        except Exception:
            existing = []

    # Merge by skill_id, preferring new entries (they have distances)
    by_id = {e["skill_id"]: e for e in existing}
    for a in activations:
        by_id[a["skill_id"]] = a
    sidecar.write_text(json.dumps(list(by_id.values())))


def _load_recorded_activation_ids(cfg: Config, session_id: str) -> set[str]:
    if not session_id:
        return set()

    sidecar = cfg.db_path.parent / "mm.activations" / f"{session_id}.json"
    if not sidecar.exists():
        return set()

    try:
        raw = json.loads(sidecar.read_text())
    except Exception:
        return set()

    skill_ids: set[str] = set()
    for entry in raw:
        if isinstance(entry, str):
            skill_ids.add(entry)
        elif isinstance(entry, dict):
            skill_id = entry.get("skill_id")
            if isinstance(skill_id, str) and skill_id:
                skill_ids.add(skill_id)
    return skill_ids


if __name__ == "__main__":
    sys.exit(main())
