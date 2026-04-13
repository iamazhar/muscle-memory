from __future__ import annotations

import json
from pathlib import Path

from muscle_memory.config import Config
from muscle_memory.debug import log_debug_event
from muscle_memory.models import Scope


def _make_config(tmp_path: Path, *, debug_enabled: bool) -> Config:
    return Config(
        db_path=tmp_path / ".claude" / "mm.db",
        scope=Scope.PROJECT,
        project_root=tmp_path,
        debug_enabled=debug_enabled,
    )


def test_log_debug_event_writes_json_line_when_enabled(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, debug_enabled=True)

    log_debug_event(
        cfg,
        component="user_prompt",
        event="no_db",
        session_id="sess-1",
        prompt_excerpt="hello",
    )

    log_path = tmp_path / ".claude" / "mm.debug.log"
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["component"] == "user_prompt"
    assert payload["event"] == "no_db"
    assert payload["session_id"] == "sess-1"
    assert payload["prompt_excerpt"] == "hello"
    assert "timestamp" in payload


def test_log_debug_event_is_noop_when_disabled(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, debug_enabled=False)

    log_debug_event(cfg, component="user_prompt", event="no_db")

    assert not (tmp_path / ".claude" / "mm.debug.log").exists()


def test_config_load_reads_mm_debug(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / ".claude" / "mm.db"
    monkeypatch.setenv("MM_DB_PATH", str(db_path))
    monkeypatch.setenv("MM_DEBUG", "1")

    cfg = Config.load(start_dir=tmp_path)

    assert cfg.debug_enabled is True
