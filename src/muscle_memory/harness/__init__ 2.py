"""Harness adapter registry."""

from __future__ import annotations

from muscle_memory.harness.base import HarnessAdapter, InstallReport
from muscle_memory.harness.claude_code import ClaudeCodeHarness
from muscle_memory.harness.codex import CodexHarness
from muscle_memory.harness.generic import GenericHarness

_HARNESSES: dict[str, HarnessAdapter] = {
    "claude-code": ClaudeCodeHarness(),
    "codex": CodexHarness(),
    "generic": GenericHarness(),
}


def harness_names() -> list[str]:
    return sorted(_HARNESSES)


def get_harness(name: str) -> HarnessAdapter:
    try:
        return _HARNESSES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_HARNESSES))
        raise ValueError(f"Unknown harness: {name!r}. Supported: {supported}") from exc


__all__ = ["HarnessAdapter", "InstallReport", "get_harness", "harness_names"]
