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

import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir

from muscle_memory.models import Scope

APP_NAME = "muscle-memory"

# Default to Sonnet 4.6 — extraction is judgment work (recurring pattern?
# non-obvious? procedural vs factual?) and smaller models anchor on
# prompt examples. Sonnet is called infrequently (once per session),
# so cost impact is small; quality compounds. Users can opt into Haiku
# via MM_LLM_MODEL=claude-haiku-4-5 for cost-sensitive setups.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_EMBEDDER = "fastembed"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMS = 384
DEFAULT_LLM_PROVIDER = "anthropic"


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

    # retrieval
    retrieval_top_k: int = 3
    retrieval_similarity_floor: float = 0.25

    # extraction
    extractor_max_skills_per_episode: int = 3

    # pool
    max_skills: int = 500

    # misc
    log_level: str = "INFO"

    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        *,
        db_path: Path | None = None,
        scope: Scope | None = None,
        start_dir: Path | None = None,
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
        # api key env var fallback per-provider
        provider_key_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(llm_provider.lower(), "ANTHROPIC_API_KEY")
        llm_api_key = _env("MM_LLM_API_KEY") or _env(provider_key_env)

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
            log_level=_env("MM_LOG_LEVEL", "INFO") or "INFO",
        )

    def ensure_db_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
