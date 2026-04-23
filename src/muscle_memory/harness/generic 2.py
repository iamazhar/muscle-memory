"""Generic harness adapter for harness-agnostic / offline-only setups."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from muscle_memory.harness.base import InstallReport
from muscle_memory.models import Trajectory

if TYPE_CHECKING:
    from muscle_memory.config import Config
    from muscle_memory.retriever import RetrievedSkill


class GenericHarness:
    name = "generic"

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
        raise NotImplementedError("generic harness has no prompt hook payload")

    def extract_session_id(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError("generic harness has no runtime session payload")

    def extract_cwd(self, payload: dict[str, Any]) -> Path | None:
        raise NotImplementedError("generic harness has no runtime cwd payload")

    def extract_transcript_path(self, payload: dict[str, Any]) -> Path | None:
        raise NotImplementedError("generic harness has no transcript payload")

    def parse_transcript(self, path: Path) -> Trajectory:
        raise NotImplementedError("generic harness does not parse harness-native transcripts")

    def format_context(self, hits: list[RetrievedSkill]) -> str:
        raise NotImplementedError("generic harness does not inject prompt context")

    def is_shell_escape(self, prompt: str) -> bool:
        return False
