"""Configuration and path resolution for muscle-memory.

Precedence, highest to lowest:
  1. Explicit values passed to `Config(...)`
  2. Environment variables (MM_*)
  3. Defaults

The DB lives at `.claude/mm.db` inside whichever project root we can find
(walking up from cwd looking for `.git` or `.claude`). If no project root
is found, or scope is "global", we fall back to a user-wide location via
platformdirs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir

from muscle_memory.models import Scope

APP_NAME = "muscle-memory"

# Default model for extraction via `claude -p`. Uses the user's
# Claude Code subscription auth, so no separate API key is needed.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_EMBEDDER = "fastembed"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMS = 384
DEFAULT_LLM_PROVIDER = "claude-code"
DEFAULT_HARNESS = "claude-code"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (or cwd) looking for `.git` or `.claude`.

    Returns the first directory that contains either, or None.
    """
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists() or (candidate / ".claude").exists():
            return candidate
    return None


def global_data_dir() -> Path:
    """User-wide data directory for muscle-memory (e.g. ~/Library/Application Support/muscle-memory)."""
    return Path(user_data_dir(APP_NAME, appauthor=False))


def project_config_path(project_root: Path | None) -> Path | None:
    """Return the project-local muscle-memory config path when available."""
    if project_root is None:
        return None
    return project_root / ".claude" / "mm.json"


def load_project_harness(project_root: Path | None) -> str | None:
    """Best-effort read of the persisted project harness."""
    path = project_config_path(project_root)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    harness = payload.get("harness")
    if isinstance(harness, str) and harness:
        return harness
    return None


def write_project_harness(project_root: Path, harness: str) -> Path:
    """Persist the selected harness for future project-local commands."""
    path = project_config_path(project_root)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"harness": harness}, indent=2) + "\n", encoding="utf-8")
    return path


@dataclass
class Config:
    """Resolved configuration for a muscle-memory session."""

    # paths
    db_path: Path
    scope: Scope
    project_root: Path | None

    # embedder
    embedder: str = DEFAULT_EMBEDDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dims: int = DEFAULT_EMBEDDING_DIMS

    # llm
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_ANTHROPIC_MODEL
    llm_api_key: str | None = None

    # harness
    harness: str = DEFAULT_HARNESS

    # retrieval
    retrieval_top_k: int = 3
    retrieval_similarity_floor: float = 0.25

    # extraction
    extractor_max_skills_per_episode: int = 3

    # pool
    max_skills: int = 500

    # misc
    log_level: str = "INFO"
    debug_enabled: bool = False
    auto_refine_enabled: bool = True

    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        *,
        db_path: Path | None = None,
        scope: Scope | None = None,
        start_dir: Path | None = None,
        harness: str | None = None,
    ) -> Config:
        """Resolve a Config from env vars and on-disk layout.

        If `db_path` is given, it wins. Otherwise `scope` decides whether
        to look for a project-local DB or the global one.
        """
        # scope
        resolved_scope: Scope
        if scope is not None:
            resolved_scope = scope
        else:
            env_scope = _env("MM_SCOPE", Scope.PROJECT.value)
            resolved_scope = Scope(str(env_scope))

        # project root detection
        project_root = find_project_root(start_dir)

        # db path
        resolved_db: Path
        if db_path is not None:
            resolved_db = db_path
        elif env_db := _env("MM_DB_PATH"):
            resolved_db = Path(env_db).expanduser()
        elif resolved_scope is Scope.PROJECT and project_root is not None:
            resolved_db = project_root / ".claude" / "mm.db"
        else:
            resolved_db = global_data_dir() / "mm.db"
            resolved_scope = Scope.GLOBAL

        # llm
        llm_provider = _env("MM_LLM_PROVIDER", DEFAULT_LLM_PROVIDER) or DEFAULT_LLM_PROVIDER
        # model default depends on provider
        if llm_provider.lower() == "openai":
            llm_model = _env("MM_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        else:
            llm_model = _env("MM_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL) or DEFAULT_ANTHROPIC_MODEL
        # api key only needed for openai provider
        llm_api_key: str | None = None
        if llm_provider.lower() == "openai":
            llm_api_key = _env("MM_LLM_API_KEY") or _env("OPENAI_API_KEY")

        project_harness = None
        if resolved_scope is Scope.PROJECT:
            project_harness = load_project_harness(project_root)
        resolved_harness = harness or _env("MM_HARNESS") or project_harness or DEFAULT_HARNESS

        # embedder
        embedder = _env("MM_EMBEDDER", DEFAULT_EMBEDDER) or DEFAULT_EMBEDDER
        embedding_model = (
            _env("MM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
        )

        return cls(
            db_path=resolved_db,
            scope=resolved_scope,
            project_root=project_root,
            embedder=embedder,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            harness=resolved_harness,
            log_level=_env("MM_LOG_LEVEL", "INFO") or "INFO",
            debug_enabled=(_env("MM_DEBUG", "0") or "0").lower() in {"1", "true", "yes", "on"},
            auto_refine_enabled=(_env("MM_AUTO_REFINE", "1") or "1").lower()
            in {"1", "true", "yes", "on"},
        )

    def ensure_db_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
