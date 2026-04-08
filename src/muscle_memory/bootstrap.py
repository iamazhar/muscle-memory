"""Seed the skill store from existing Claude Code session history.

Claude Code stores session JSONLs under `~/.claude/projects/<encoded-path>/`.
We walk those files, parse each into a Trajectory, infer outcomes,
and run the extractor to produce candidate Skills.

This lets new users get value from muscle-memory on day one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.embeddings import Embedder
from muscle_memory.extractor import ExtractionError, Extractor
from muscle_memory.hooks.stop import parse_transcript
from muscle_memory.llm import LLM
from muscle_memory.models import Episode, Skill
from muscle_memory.outcomes import infer_outcome


@dataclass
class BootstrapReport:
    sessions_considered: int = 0
    sessions_parsed: int = 0
    episodes_added: int = 0
    skills_extracted: int = 0
    errors: list[str] = field(default_factory=list)
    aborted_reason: str | None = None


def find_session_files(
    *,
    project_path: Path | None = None,
    since: datetime | None = None,
    claude_home: Path | None = None,
) -> list[Path]:
    """Return JSONL session files, optionally scoped to a project and time window.

    Claude Code encodes project paths in the directory name by replacing
    slashes with dashes. We detect that encoding if `project_path` is
    given, otherwise we pick up every project.
    """
    root = claude_home or (Path.home() / ".claude" / "projects")
    if not root.exists():
        return []

    if project_path is not None:
        encoded = str(project_path.resolve()).replace("/", "-")
        if encoded.startswith("-"):
            encoded = encoded  # already starts with - for absolute paths
        candidates = [root / encoded]
        # also try with leading dash (some versions include it)
        alt = root / ("-" + encoded if not encoded.startswith("-") else encoded[1:])
        candidates.append(alt)
        dirs = [p for p in candidates if p.exists() and p.is_dir()]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]

    files: list[Path] = []
    for d in dirs:
        for jsonl in d.glob("*.jsonl"):
            if since is not None:
                mtime = datetime.fromtimestamp(jsonl.stat().st_mtime, tz=UTC)
                if mtime < since:
                    continue
            files.append(jsonl)

    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def bootstrap(
    *,
    config: Config,
    store: Store,
    embedder: Embedder,
    llm: LLM,
    days: int | None = 30,
    project_only: bool = True,
    max_sessions: int = 200,
) -> BootstrapReport:
    report = BootstrapReport()

    since: datetime | None = None
    if days is not None:
        since = datetime.now(UTC) - timedelta(days=days)

    project_path = config.project_root if project_only else None
    files = find_session_files(
        project_path=project_path,
        since=since,
    )
    files = files[-max_sessions:]

    extractor = Extractor(llm, config)

    for path in files:
        report.sessions_considered += 1
        try:
            trajectory = parse_transcript(path)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"{path.name}: parse failed: {e}")
            continue

        if not trajectory.tool_calls:
            continue
        report.sessions_parsed += 1

        # Bootstrap has no activation sidecars — it's processing historical
        # sessions from before muscle-memory was installed. Falls back to
        # keyword-only heuristic.
        signal = infer_outcome(trajectory)

        episode = Episode(
            session_id=path.stem,
            user_prompt=trajectory.user_prompt or "(bootstrap)",
            trajectory=trajectory,
            outcome=signal.outcome,
            reward=signal.reward,
            project_path=str(config.project_root) if config.project_root else None,
        )
        try:
            store.add_episode(episode)
            report.episodes_added += 1
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"{path.name}: episode insert failed: {e}")
            continue

        try:
            skills = extractor.extract(episode)
        except ExtractionError as e:
            # Surface LLM errors loudly. If the first one fails for
            # auth/credit/quota reasons, abort — every subsequent call
            # will fail the same way and waste time.
            msg = str(e)
            report.errors.append(f"{path.name}: extraction failed: {msg}")
            if _looks_fatal(msg) or report.skills_extracted == 0:
                report.aborted_reason = msg
                return report
            continue

        for skill in skills:
            if _too_similar(store, skill, embedder):
                continue
            embedding = embedder.embed_one(skill.activation)
            try:
                store.add_skill(skill, embedding=embedding)
                report.skills_extracted += 1
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"add skill failed: {e}")

    return report


def _looks_fatal(msg: str) -> bool:
    """Heuristic: does this LLM error look like it'll keep happening?"""
    low = msg.lower()
    fatal_markers = (
        "credit",
        "quota",
        "billing",
        "unauthorized",
        "authentication",
        "api key",
        "not found",  # wrong model name
        "invalid_request_error",
    )
    return any(m in low for m in fatal_markers)


def _too_similar(store: Store, skill: Skill, embedder: Embedder) -> bool:
    """Crude dedup: if any existing skill has very similar activation, skip."""
    emb = embedder.embed_one(skill.activation)
    hits = store.search_skills_by_embedding(emb, top_k=1)
    if not hits:
        return False
    _existing, distance = hits[0]
    return distance < 0.1  # very close match
