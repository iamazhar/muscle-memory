"""Harness-aware install/uninstall entrypoints for runtime integration."""

from __future__ import annotations

from pathlib import Path

from muscle_memory.config import Config, write_project_harness
from muscle_memory.harness import InstallReport, get_harness, harness_names
from muscle_memory.models import Scope


def install(
    *,
    scope: Scope = Scope.PROJECT,
    project_root: Path | None = None,
    harness: str | None = None,
) -> InstallReport:
    """Initialize the project DB and install harness-specific runtime integration."""
    cfg = Config.load(scope=scope, start_dir=project_root, harness=harness)
    if scope is Scope.PROJECT and cfg.project_root is None:
        raise RuntimeError(
            "Not inside a project (no .git or .claude found). Either `cd` into a project or use --scope global."
        )

    for other_name in harness_names():
        if other_name == cfg.harness:
            continue
        get_harness(other_name).uninstall(cfg)

    adapter = get_harness(cfg.harness)
    report = adapter.install(cfg)
    if scope is Scope.PROJECT and cfg.project_root is not None:
        write_project_harness(cfg.project_root, cfg.harness)
    return report


def uninstall(
    *,
    project_root: Path | None = None,
    harness: str | None = None,
) -> InstallReport:
    """Remove harness-specific runtime integration. The DB is preserved."""
    cfg = Config.load(start_dir=project_root, harness=harness)
    adapter = get_harness(cfg.harness)
    return adapter.uninstall(cfg)
