"""Codex transcript adapter for offline transcript ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from muscle_memory.harness.base import InstallReport
from muscle_memory.models import ToolCall, Trajectory

if TYPE_CHECKING:
    from muscle_memory.config import Config
    from muscle_memory.retriever import RetrievedSkill


class CodexHarness:
    name = "codex"

    def install(self, config: Config) -> InstallReport:
        config.ensure_db_dir()
        from muscle_memory.db import Store

        Store(config.db_path, embedding_dims=config.embedding_dims)
        return InstallReport(
            settings_path=None,
            db_path=config.db_path,
            installed_events=[],
            already_present=[],
        )

    def uninstall(self, config: Config) -> InstallReport:
        return InstallReport(
            settings_path=None,
            db_path=config.db_path,
            installed_events=[],
            already_present=[],
        )

    def extract_prompt(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError("codex harness does not yet support runtime prompt hooks")

    def extract_session_id(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError("codex harness does not yet support runtime session hooks")

    def extract_cwd(self, payload: dict[str, Any]) -> Path | None:
        raise NotImplementedError("codex harness does not yet support runtime cwd hooks")

    def extract_transcript_path(self, payload: dict[str, Any]) -> Path | None:
        raise NotImplementedError("codex harness does not yet support runtime transcript hooks")

    def parse_transcript(self, path: Path) -> Trajectory:
        assistant_turns: list[str] = []
        tool_calls: list[ToolCall] = []

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("type") != "item.completed":
                    continue
                item = record.get("item")
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str):
                        normalized = text.strip()
                        if normalized:
                            assistant_turns.append(normalized)
                    continue

                if item_type == "command_execution":
                    command = item.get("command")
                    if not isinstance(command, str) or not command:
                        continue
                    aggregated_output = item.get("aggregated_output")
                    output = aggregated_output if isinstance(aggregated_output, str) else ""
                    exit_code = item.get("exit_code")
                    tool_call = ToolCall(name="Bash", arguments={"command": command})
                    if isinstance(exit_code, int) and exit_code != 0:
                        tool_call.error = output or f"command failed with exit {exit_code}"
                    else:
                        tool_call.result = output
                    tool_calls.append(tool_call)

        return Trajectory(
            user_prompt="",
            tool_calls=tool_calls,
            assistant_turns=assistant_turns,
        )

    def format_context(self, hits: list[RetrievedSkill]) -> str:
        raise NotImplementedError("codex harness does not yet inject prompt context automatically")

    def is_shell_escape(self, prompt: str) -> bool:
        return False
