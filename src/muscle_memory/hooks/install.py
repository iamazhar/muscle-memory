"""Auto-wire muscle-memory hooks into Claude Code's `.claude/settings.json`.

Called by `mm init`. Safe to run multiple times — idempotent and
non-destructive to other hooks.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from muscle_memory.config import Config
from muscle_memory.models import Scope

# Claude Code hook events we wire.
USER_PROMPT_EVENT = "UserPromptSubmit"
STOP_EVENT = "Stop"

USER_PROMPT_CMD = "mm hook user-prompt"
STOP_CMD = "mm hook stop"


@dataclass
class InstallReport:
    settings_path: Path
    db_path: Path
    installed_events: list[str]
    already_present: list[str]


def install(
    *,
    scope: Scope = Scope.PROJECT,
    project_root: Path | None = None,
) -> InstallReport:
    """Wire the hooks into settings.json and ensure the DB exists."""
    cfg = Config.load(scope=scope, start_dir=project_root)

    if scope is Scope.PROJECT and cfg.project_root is None:
        raise RuntimeError(
            "Not inside a project (no .git or .claude found). "
            "Either `cd` into a project or use --scope global."
        )

    cfg.ensure_db_dir()

    # just initialize the store so the schema is ready
    from muscle_memory.db import Store

    Store(cfg.db_path, embedding_dims=cfg.embedding_dims)

    settings_root = cfg.project_root if scope is Scope.PROJECT else Path.home()
    assert settings_root is not None
    settings_path = settings_root / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Could not parse existing {settings_path}. "
                "Fix it manually or remove it before running `mm init`."
            ) from None

    original = deepcopy(settings)

    installed: list[str] = []
    already: list[str] = []

    for event, cmd in ((USER_PROMPT_EVENT, USER_PROMPT_CMD), (STOP_EVENT, STOP_CMD)):
        changed = _ensure_hook(settings, event, cmd)
        if changed:
            installed.append(event)
        else:
            already.append(event)

    if settings != original:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    return InstallReport(
        settings_path=settings_path,
        db_path=cfg.db_path,
        installed_events=installed,
        already_present=already,
    )


def _ensure_hook(settings: dict[str, Any], event: str, command: str) -> bool:
    """Add `command` to the given Claude Code `event` hook if not already present.

    Returns True if a modification was made, False if it was already there.
    """
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise RuntimeError(".claude/settings.json 'hooks' must be an object")

    event_hooks = hooks.get(event)
    if event_hooks is None:
        event_hooks = []
        hooks[event] = event_hooks

    if not isinstance(event_hooks, list):
        raise RuntimeError(f".claude/settings.json hooks.{event} must be a list")

    # Claude Code hook shape:
    # {
    #   "matcher": "",
    #   "hooks": [
    #     { "type": "command", "command": "..." }
    #   ]
    # }
    for group in event_hooks:
        if not isinstance(group, dict):
            continue
        for h in group.get("hooks", []) or []:
            if isinstance(h, dict) and h.get("command") == command:
                return False

    event_hooks.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": command}],
        }
    )
    return True


def uninstall(*, project_root: Path | None = None) -> InstallReport:
    """Remove muscle-memory hooks from settings.json. DB is preserved."""
    cfg = Config.load(start_dir=project_root)
    settings_root = cfg.project_root if cfg.project_root else Path.home()
    settings_path = settings_root / ".claude" / "settings.json"

    removed: list[str] = []

    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        hooks = settings.get("hooks", {})
        for event in (USER_PROMPT_EVENT, STOP_EVENT):
            groups = hooks.get(event, []) or []
            filtered: list[dict[str, Any]] = []
            for group in groups:
                if not isinstance(group, dict):
                    filtered.append(group)
                    continue
                inner = [
                    h
                    for h in (group.get("hooks") or [])
                    if not (isinstance(h, dict) and h.get("command", "").startswith("mm hook"))
                ]
                if inner:
                    group["hooks"] = inner
                    filtered.append(group)
                else:
                    removed.append(event)
            if filtered:
                hooks[event] = filtered
            elif event in hooks:
                del hooks[event]

        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    return InstallReport(
        settings_path=settings_path,
        db_path=cfg.db_path,
        installed_events=[],
        already_present=removed,
    )
