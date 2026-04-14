"""Best-effort debug logging for hooks and background jobs.

Logging must never break the main workflow. All helpers in this module
silently swallow filesystem/serialization failures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from muscle_memory.config import Config


def log_debug_event(config: Config, *, component: str, event: str, **details: Any) -> None:
    """Append one structured debug event when debug mode is enabled.

    Writes newline-delimited JSON next to the project's DB under
    `.claude/mm.debug.log`. Failures are intentionally ignored.
    """
    if not getattr(config, "debug_enabled", False):
        return

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "component": component,
        "event": event,
        **details,
    }

    try:
        if config.project_root is not None:
            log_path = config.project_root / ".claude" / "mm.debug.log"
        else:
            log_path = config.db_path.parent / "mm.debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
