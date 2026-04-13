"""Harness adapter interfaces for runtime integration and transcript parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from muscle_memory.config import Config
    from muscle_memory.models import Trajectory
    from muscle_memory.retriever import RetrievedSkill


@dataclass
class InstallReport:
    settings_path: Path | None
    db_path: Path
    installed_events: list[str]
    already_present: list[str]


@runtime_checkable
class HarnessAdapter(Protocol):
    name: str

    def install(self, config: Config) -> InstallReport: ...

    def uninstall(self, config: Config) -> InstallReport: ...

    def extract_prompt(self, payload: dict[str, Any]) -> str: ...

    def extract_session_id(self, payload: dict[str, Any]) -> str: ...

    def extract_cwd(self, payload: dict[str, Any]) -> Path | None: ...

    def extract_transcript_path(self, payload: dict[str, Any]) -> Path | None: ...

    def parse_transcript(self, path: Path) -> Trajectory: ...

    def format_context(self, hits: list[RetrievedSkill]) -> str: ...

    def is_shell_escape(self, prompt: str) -> bool: ...
