"""`mm` — command-line interface for muscle-memory."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from muscle_memory import __version__
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.embeddings import make_embedder
from muscle_memory.eval.benchmark import _current_source_tree_sha256
from muscle_memory.models import (
    BackgroundJob,
    DeliveryMode,
    JobKind,
    JobStatus,
    Maturity,
    Outcome,
    Scope,
    Skill,
)

console = Console()
app = typer.Typer(
    name="mm",
    help="muscle-memory: practiced skill for coding agents.",
    no_args_is_help=True,
    add_completion=False,
)

maint_app = typer.Typer(help="Advanced maintenance, repair, and runtime controls.")
app.add_typer(maint_app, name="maint", hidden=True)

share_app = typer.Typer(help="Advanced import/export for skills.")
app.add_typer(share_app, name="share", hidden=True)

review_app = typer.Typer(help="Advanced review for quarantined candidate skills.")
app.add_typer(review_app, name="review", hidden=True)

jobs_app = typer.Typer(help="Advanced inspection for background jobs.")
app.add_typer(jobs_app, name="jobs", hidden=True)

ingest_app = typer.Typer(help="Advanced transcript and episode ingestion.")
app.add_typer(ingest_app, name="ingest", hidden=True)

hook_app = typer.Typer(help="Claude Code hook handlers (not for direct use).")
app.add_typer(hook_app, name="hook", hidden=True)

eval_app = typer.Typer(help="Advanced evaluation for outcome detection and skill impact.")
app.add_typer(eval_app, name="eval", hidden=True)

simulate_app = typer.Typer(help="Advanced synthetic dogfooding without real sessions.")
app.add_typer(simulate_app, name="simulate", hidden=True)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _load_config(scope: Scope | None = None) -> Config:
    return Config.load(scope=scope)


def _stdin_is_interactive() -> bool:
    return sys.stdin.isatty()


def _validate_init_scope(scope: Scope) -> None:
    if scope is not Scope.PROJECT:
        return
    cfg = Config.load(scope=scope)
    if cfg.project_root is None:
        raise RuntimeError(
            "Not inside a project (no .git or .claude found). Either `cd` into a project or use --scope global."
        )


def _resolve_init_harness(harness: str | None) -> str:
    if harness:
        return harness
    if not _stdin_is_interactive():
        raise RuntimeError(
            "Pass --harness claude-code|codex|generic when running `mm init` non-interactively."
        )
    return Prompt.ask(
        "Choose a harness",
        choices=["claude-code", "codex"],
        default="claude-code",
        console=console,
    )


def _open_store(cfg: Config) -> Store:
    if not cfg.db_path.exists():
        console.print(
            f"[red]No muscle-memory database at {cfg.db_path}.[/red]\n"
            "Run [bold]mm init[/bold] inside a project first."
        )
        raise typer.Exit(2)
    return Store(cfg.db_path, embedding_dims=cfg.embedding_dims)


def _current_repo_head(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _current_worktree_state(repo_root: Path | None) -> tuple[bool | None, str | None]:
    if repo_root is None:
        return None, None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    status_lines = [line for line in result.stdout.splitlines() if line[3:] != "benchmark-run.json"]
    status = "\n".join(status_lines)
    if status_lines:
        status += "\n"
    return status == "", hashlib.sha256(status.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_maturity(m: Maturity) -> str:
    colors = {
        Maturity.CANDIDATE: "yellow",
        Maturity.LIVE: "cyan",
        Maturity.PROVEN: "green",
    }
    return f"[{colors[m]}]{m.value}[/{colors[m]}]"


def _format_score(score: float) -> str:
    if score >= 0.7:
        return f"[green]{score:.2f}[/green]"
    if score >= 0.4:
        return f"[yellow]{score:.2f}[/yellow]"
    if score == 0.0:
        return f"[dim]{score:.2f}[/dim]"
    return f"[red]{score:.2f}[/red]"


def _format_outcome(outcome: Outcome) -> str:
    colors = {
        Outcome.SUCCESS: "green",
        Outcome.FAILURE: "red",
        Outcome.UNKNOWN: "dim",
    }
    return f"[{colors[outcome]}]{outcome.value}[/{colors[outcome]}]"


def _short_id(s: str, n: int = 8) -> str:
    return s[:n]


def _ensure_utc(dt: datetime) -> datetime:
    """Coerce a datetime to UTC. Naive datetimes are assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _relative_time(dt: datetime) -> str:
    """Format a datetime as a human-readable relative time string."""
    delta = datetime.now(UTC) - _ensure_utc(dt)
    if delta.days > 30:
        months = delta.days // 30
        return f"{months}mo ago"
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    if minutes > 0:
        return f"{minutes}m ago"
    return "just now"


# ----------------------------------------------------------------------
# top-level commands
# ----------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"muscle-memory {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """muscle-memory: practiced skill for coding agents."""


@app.command(hidden=True)
def version() -> None:
    """Print version and exit."""
    console.print(f"muscle-memory {__version__}")


@maint_app.command("pause")
@app.command(hidden=True)
def pause() -> None:
    """Pause muscle-memory. Hooks will silently no-op until resumed."""
    cfg = _load_config()
    flag = cfg.db_path.parent / "mm.paused"
    flag.touch()
    console.print(
        "[yellow]muscle-memory paused.[/yellow] No retrieval or extraction until `mm resume`."
    )


@maint_app.command("resume")
@app.command(hidden=True)
def resume() -> None:
    """Resume muscle-memory after a pause."""
    cfg = _load_config()
    flag = cfg.db_path.parent / "mm.paused"
    if flag.is_file():
        flag.unlink()
        console.print("[green]muscle-memory resumed.[/green]")
    else:
        console.print("[dim]muscle-memory is not paused.[/dim]")


@app.command()
def init(
    scope: Scope = typer.Option(
        Scope.PROJECT,
        "--scope",
        "-s",
        help="project (default) or global",
    ),
    harness: str | None = typer.Option(
        None,
        "--harness",
        help="Runtime harness to integrate with. Omit in a terminal to choose interactively.",
    ),
) -> None:
    """Set up muscle-memory for the current project.

    Creates `.claude/mm.db` and installs harness-specific runtime integration when supported.
    """
    from muscle_memory.hooks.install import install as do_install

    try:
        _validate_init_scope(scope)
        selected_harness = _resolve_init_harness(harness)
        report = do_install(scope=scope, harness=selected_harness)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    settings_display = (
        str(report.settings_path) if report.settings_path is not None else "(not used)"
    )
    if selected_harness == "claude-code":
        next_step = (
            "Next: use Claude Code as usual. Optionally seed with [bold]mm bootstrap[/bold]."
        )
    elif selected_harness == "codex":
        next_step = (
            "Next: use Codex with [bold]mm retrieve[/bold] or transcript ingestion. "
            "Automatic prompt hooks are not installed for Codex yet."
        )
    else:
        next_step = "Next: ingest transcripts or use [bold]mm retrieve[/bold] from your harness/orchestrator."

    console.print(
        Panel.fit(
            f"[green]muscle-memory initialized[/green]\n\n"
            f"Harness: [bold]{selected_harness}[/bold]\n"
            f"DB: [bold]{report.db_path}[/bold]\n"
            f"Settings: [bold]{settings_display}[/bold]\n"
            f"Installed hooks: {', '.join(report.installed_events) or '(none)'}\n"
            f"Already present: {', '.join(report.already_present) or '—'}",
            title="init complete",
        )
    )
    console.print(next_step)


@app.command("skills")
@app.command("list", hidden=True)
def list_skills(
    maturity: Maturity | None = typer.Option(None, "--maturity", "-m"),
    limit: int = typer.Option(50, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List the skills learned for this project."""
    cfg = _load_config()
    store = _open_store(cfg)
    skills = store.list_skills(maturity=maturity, limit=limit)

    if as_json:
        typer.echo(json.dumps([_skill_to_dict(s) for s in skills], indent=2))
        return

    if not skills:
        console.print(
            "[dim]No skills yet. Try [bold]mm bootstrap[/bold] to seed from history.[/dim]"
        )
        return

    table = Table(title=f"muscle-memory skills ({len(skills)})")
    table.add_column("id", style="dim", width=10)
    table.add_column("maturity", width=12)
    table.add_column("score", justify="right")
    table.add_column("uses", justify="right")
    table.add_column("activation")

    for s in skills:
        table.add_row(
            _short_id(s.id),
            _format_maturity(s.maturity),
            _format_score(s.score),
            f"{s.successes}/{s.invocations}",
            s.activation,
        )
    console.print(table)


@app.command()
def show(skill_id: str = typer.Argument(..., help="Skill id or prefix.")) -> None:
    """Show a single skill in full."""
    cfg = _load_config()
    store = _open_store(cfg)
    skill = _resolve_skill(store, skill_id)

    console.print(
        Panel(
            (
                f"[bold]activation[/bold]\n{skill.activation}\n\n"
                f"[bold]execution[/bold]\n{skill.execution}\n\n"
                f"[bold]termination[/bold]\n{skill.termination}\n\n"
                f"[dim]maturity:[/dim] {_format_maturity(skill.maturity)}   "
                f"[dim]score:[/dim] {_format_score(skill.score)}   "
                f"[dim]uses:[/dim] {skill.successes}/{skill.invocations}\n"
                f"[dim]tags:[/dim] {', '.join(skill.tags) or '—'}\n"
                f"[dim]tool hints:[/dim] {', '.join(skill.tool_hints) or '—'}"
            ),
            title=f"skill {_short_id(skill.id)}",
        )
    )


@app.command("use")
def use_skill(
    prompt: str = typer.Argument(..., help="Task to get practiced context for."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Use practiced skill for a task."""
    from muscle_memory.personal_loop import (
        capture_task,
        count_text_tokens,
        format_context,
        record_activations,
    )
    from muscle_memory.retriever import Retriever

    cfg = _load_config()
    store = _open_store(cfg)
    task = capture_task(
        store,
        raw_prompt=prompt,
        harness=cfg.harness,
        project_path=str(cfg.project_root) if cfg.project_root is not None else None,
    )
    embedder = make_embedder(cfg)
    hits = Retriever(store, embedder, cfg).retrieve(task.cleaned_prompt)
    context = format_context(hits)
    context_token_count = count_text_tokens(context)
    activations = record_activations(
        store,
        task=task,
        hits=hits,
        delivery_mode=DeliveryMode.CODEX_USE if cfg.harness == "codex" else DeliveryMode.MANUAL,
        context_token_count=context_token_count,
    )

    if as_json:
        payload = {
            "task": {
                "id": task.id,
                "cleaned_prompt": task.cleaned_prompt,
                "harness": task.harness,
            },
            "context": context,
            "context_token_count": context_token_count,
            "hits": [
                {
                    **_skill_to_dict(hit.skill),
                    "distance": hit.distance,
                    "final_rank": hit.final_rank,
                    "activation_id": activations[index].id,
                    "activation": {
                        "id": activations[index].id,
                        "delivery_mode": activations[index].delivery_mode.value,
                        "injected_token_count": activations[index].injected_token_count,
                    },
                }
                for index, hit in enumerate(hits)
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not hits:
        console.print("[dim]No matching skills. Proceed normally.[/dim]")
        return
    console.print(context)


@app.command(hidden=True)
def retrieve(
    prompt: str = typer.Argument(..., help="Prompt/task to retrieve relevant skills for."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Retrieve the practiced skills relevant to a task."""
    from muscle_memory.retriever import Retriever

    cfg = _load_config()
    store = _open_store(cfg)
    embedder = make_embedder(cfg)
    hits = Retriever(store, embedder, cfg).retrieve(prompt)

    if as_json:
        payload = []
        for hit in hits:
            item = _skill_to_dict(hit.skill)
            item["distance"] = hit.distance
            item["score_bonus"] = hit.score_bonus
            payload.append(item)
        typer.echo(json.dumps(payload, indent=2))
        return

    if not hits:
        console.print("[dim]No matching skills.[/dim]")
        return

    table = Table(title=f"retrieved skills ({len(hits)})")
    table.add_column("id", style="dim", width=10)
    table.add_column("maturity", width=12)
    table.add_column("distance", justify="right")
    table.add_column("activation")
    for hit in hits:
        table.add_row(
            _short_id(hit.skill.id),
            _format_maturity(hit.skill.maturity),
            f"{hit.distance:.3f}",
            hit.skill.activation,
        )
    console.print(table)


@review_app.command("list")
def review_list(
    limit: int = typer.Option(50, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List quarantined candidate skills awaiting review or promotion."""
    cfg = _load_config()
    store = _open_store(cfg)
    skills = store.list_skills(maturity=Maturity.CANDIDATE, limit=limit)
    skills.sort(
        key=lambda s: (len(dict.fromkeys(s.source_episode_ids)), s.created_at),
        reverse=True,
    )

    if as_json:
        payload = []
        for s in skills:
            item = _skill_to_dict(s)
            item.update(_candidate_review_metadata(s))
            payload.append(item)
        typer.echo(json.dumps(payload, indent=2))
        return

    if not skills:
        console.print("[green]No candidate skills awaiting review.[/green]")
        return

    table = Table(title=f"candidate skills ({len(skills)})")
    table.add_column("id", style="dim", width=10)
    table.add_column("evidence", justify="right", width=8)
    table.add_column("created", width=10)
    table.add_column("reason")
    table.add_column("activation")

    for s in skills:
        meta = _candidate_review_metadata(s)
        table.add_row(
            _short_id(s.id),
            str(meta["source_evidence"]),
            _relative_time(s.created_at),
            str(meta["review_reason"]),
            s.activation,
        )
    console.print(table)


@review_app.command("approve")
def review_approve(skill_id: str = typer.Argument(..., help="Skill id or prefix.")) -> None:
    """Promote a candidate so it becomes retrievable."""
    cfg = _load_config()
    store = _open_store(cfg)
    skill = _resolve_skill(store, skill_id)

    if skill.maturity is Maturity.PROVEN:
        console.print("[dim]Skill is already proven.[/dim]")
        return
    if skill.maturity is not Maturity.LIVE:
        skill.maturity = Maturity.LIVE
        store.update_skill(skill)

    console.print(f"[green]Approved[/green] {skill.id[:8]} as [cyan]{skill.maturity.value}[/cyan].")


@review_app.command("reject")
def review_reject(skill_id: str = typer.Argument(..., help="Skill id or prefix.")) -> None:
    """Delete a candidate judged to be junk or not worth keeping."""
    cfg = _load_config()
    store = _open_store(cfg)
    skill = _resolve_skill(store, skill_id)
    store.delete_skill(skill.id)
    console.print(f"[green]Rejected and deleted {skill.id[:8]}.[/green]")


@jobs_app.command("list")
def jobs_list(
    status: JobStatus | None = typer.Option(None, "--status", help="Filter by job status."),
    kind: JobKind | None = typer.Option(None, "--kind", help="Filter by job kind."),
    limit: int = typer.Option(50, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List tracked background jobs."""
    cfg = _load_config()
    store = _open_store(cfg)
    jobs = store.list_jobs(limit=limit, status=status, kind=kind)

    if as_json:
        typer.echo(json.dumps([_job_to_dict(j) for j in jobs], indent=2))
        return

    if not jobs:
        console.print("[dim]No tracked jobs.[/dim]")
        return

    table = Table(title=f"tracked jobs ({len(jobs)})")
    table.add_column("id", style="dim", width=10)
    table.add_column("kind", width=10)
    table.add_column("status", width=10)
    table.add_column("attempts", justify="right")
    table.add_column("updated")
    table.add_column("error")

    for job in jobs:
        table.add_row(
            _short_id(job.id),
            job.kind.value,
            job.status.value,
            str(job.attempts),
            _relative_time(job.updated_at),
            (job.error or "")[:60],
        )
    console.print(table)


@jobs_app.command("retry")
def jobs_retry(job_id: str = typer.Argument(..., help="Job id or prefix.")) -> None:
    """Retry a failed tracked background job."""
    cfg = _load_config()
    store = _open_store(cfg)
    job = _resolve_job(store, job_id)

    if job.status is not JobStatus.FAILED:
        console.print(f"[yellow]Job {job.id[:8]} is {job.status.value}, not failed.[/yellow]")
        raise typer.Exit(1)

    store.update_job_status(job.id, status=JobStatus.PENDING, attempts=job.attempts + 1, error=None)
    updated = store.get_job(job.id)
    assert updated is not None
    _spawn_job_retry(updated, cfg)
    console.print(f"[green]Requeued {updated.kind.value} job {updated.id[:8]}.[/green]")


@jobs_app.command("retry-failed")
def jobs_retry_failed() -> None:
    """Retry all failed tracked jobs."""
    cfg = _load_config()
    store = _open_store(cfg)
    failed_jobs = store.list_jobs(limit=None, status=JobStatus.FAILED)
    if not failed_jobs:
        console.print("[green]No failed jobs to retry.[/green]")
        return

    for job in failed_jobs:
        store.update_job_status(
            job.id, status=JobStatus.PENDING, attempts=job.attempts + 1, error=None
        )
        updated = store.get_job(job.id)
        assert updated is not None
        _spawn_job_retry(updated, cfg)

    console.print(f"[green]Requeued {len(failed_jobs)} failed job(s).[/green]")


@jobs_app.command("delete")
def jobs_delete(
    job_id: str = typer.Argument(..., help="Job id or prefix."),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip the confirmation prompt for non-failed jobs.",
    ),
) -> None:
    """Delete a tracked background job. Useful for jobs whose underlying
    episode or context is gone — e.g. auth-failed extractions that will
    never succeed on retry."""
    cfg = _load_config()
    store = _open_store(cfg)
    job = _resolve_job(store, job_id)

    # Only failed jobs are safe to delete silently. Anything else risks
    # losing work in flight (pending/running) or audit history (succeeded).
    if job.status is not JobStatus.FAILED and not force:
        console.print(
            f"[yellow]Refusing to delete {job.status.value} job "
            f"{job.id[:8]}. Re-run with --force to override.[/yellow]"
        )
        raise typer.Exit(1)

    removed = store.delete_job(job.id)
    if removed:
        console.print(f"[green]Deleted {job.kind.value} job {job.id[:8]}.[/green]")
    else:
        console.print(f"[red]Job {job.id[:8]} could not be deleted.[/red]")
        raise typer.Exit(1)


@jobs_app.command("purge-failed")
def jobs_purge_failed(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the interactive confirmation."),
) -> None:
    """Delete every failed tracked job. For the 'mark dead' workflow when
    retries are not worth attempting (e.g. permanent auth errors)."""
    cfg = _load_config()
    store = _open_store(cfg)
    failed_jobs = store.list_jobs(limit=None, status=JobStatus.FAILED)
    if not failed_jobs:
        console.print("[green]No failed jobs to purge.[/green]")
        return

    if not yes:
        console.print(f"[yellow]About to delete {len(failed_jobs)} failed job(s).[/yellow]")
        confirm = typer.confirm("Proceed?", default=False)
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit(1)

    removed = store.delete_jobs_by_status(JobStatus.FAILED)
    console.print(f"[green]Purged {removed} failed job(s).[/green]")


def _spawn_job_retry(job: BackgroundJob, cfg: Config) -> None:
    from muscle_memory.hooks.stop import _fire_async_extraction, _fire_async_refinement

    if job.kind is JobKind.EXTRACT:
        episode_id = str(job.payload.get("episode_id", ""))
        if not episode_id:
            raise RuntimeError("extract job missing episode_id")
        _fire_async_extraction(episode_id, cfg.db_path, job_id=job.id)
        return
    if job.kind is JobKind.REFINE:
        _fire_async_refinement(cfg.db_path, job_id=job.id)
        return
    raise RuntimeError(f"unsupported job kind: {job.kind.value}")


def log(
    outcome: Outcome | None = typer.Option(None, "--outcome", "-o", help="Filter by outcome."),
    limit: int = typer.Option(20, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Show recent episodes with outcomes."""
    cfg = _load_config()
    store = _open_store(cfg)
    # Over-fetch when filtering so we get enough matching results.
    fetch_limit = limit * 5 if outcome is not None else limit
    episodes = store.list_episodes(limit=fetch_limit)

    if outcome is not None:
        episodes = [ep for ep in episodes if ep.outcome == outcome][:limit]

    if as_json:
        data = []
        for ep in episodes:
            data.append(
                {
                    "id": _short_id(ep.id),
                    "session_id": ep.session_id or "",
                    "prompt": ep.user_prompt[:80],
                    "outcome": ep.outcome.value,
                    "reward": ep.reward,
                    "tool_calls": ep.trajectory.num_tool_calls(),
                    "skills_activated": len(ep.activated_skills),
                    "started_at": ep.started_at.isoformat() if ep.started_at else None,
                }
            )
        typer.echo(json.dumps(data, indent=2))
        return

    if not episodes:
        console.print("[dim]No episodes recorded yet.[/dim]")
        return

    table = Table(title=f"recent episodes ({len(episodes)})")
    table.add_column("time", width=10)
    table.add_column("prompt")
    table.add_column("outcome", width=10)
    table.add_column("reward", justify="right", width=7)
    table.add_column("tools", justify="right", width=6)
    table.add_column("skills", justify="right", width=6)

    for ep in episodes:
        table.add_row(
            _relative_time(ep.started_at) if ep.started_at else "—",
            ep.user_prompt[:60] + ("…" if len(ep.user_prompt) > 60 else ""),
            _format_outcome(ep.outcome),
            f"{ep.reward:+.1f}",
            str(ep.trajectory.num_tool_calls()),
            str(len(ep.activated_skills)),
        )
    console.print(table)


@app.command("status")
@app.command("stats", hidden=True)
def stats(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Show whether muscle-memory is improving outcomes and reuse."""
    from muscle_memory.personal_loop import compute_proof_metrics

    cfg = _load_config()
    store = _open_store(cfg)
    skills = store.list_skills()
    episodes = store.list_episodes(limit=1000)
    episodes_total = store.count_episodes()

    now = datetime.now(UTC)
    paused = (cfg.db_path.parent / "mm.paused").exists()

    # -- Compute all metrics up-front --

    # Maturity breakdown
    by_maturity: dict[Maturity, int] = {m: 0 for m in Maturity}
    for s in skills:
        by_maturity[s.maturity] += 1

    # Reuse rate
    total_invocations = sum(s.invocations for s in skills)
    total_successes = sum(s.successes for s in skills)
    effective_total_invocations = sum(max(s.invocations, s.successes + s.failures) for s in skills)
    reuse_rate = (
        total_successes / effective_total_invocations if effective_total_invocations else 0.0
    )

    # Episodes with skills + avg reward
    eps_with_skills = [ep for ep in episodes if ep.activated_skills]
    rewards = [ep.reward for ep in eps_with_skills]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # Learning metrics
    cutoff_7d = now - timedelta(days=7)
    new_7d = sum(1 for s in skills if _ensure_utc(s.created_at) >= cutoff_7d)
    last_learned_at = max((s.created_at for s in skills), default=None)
    refined_skills = [s for s in skills if s.refinement_count > 0]
    total_refinements = sum(s.refinement_count for s in refined_skills)

    # Attention metrics
    from muscle_memory.eval.evaluator import evaluate_governance
    from muscle_memory.refine import should_auto_refine

    need_refine = [s for s in skills if should_auto_refine(s)]
    at_risk = [s for s in skills if s.invocations >= 5 and s.score <= 0.2]
    inconsistent_counters = [s for s in skills if s.successes + s.failures > s.invocations]
    pending_review = [s for s in skills if s.maturity is Maturity.CANDIDATE]
    cutoff_30d = now - timedelta(days=30)
    stale = [
        s
        for s in skills
        if s.invocations > 0
        and s.last_used_at is not None
        and _ensure_utc(s.last_used_at) < cutoff_30d
    ]
    unknown_count = sum(1 for ep in episodes if ep.outcome is Outcome.UNKNOWN)
    unknown_rate = (unknown_count / len(episodes)) if episodes else 0.0
    jobs = store.list_jobs(limit=None)
    pending_jobs = sum(1 for job in jobs if job.status in {JobStatus.PENDING, JobStatus.RUNNING})
    failed_jobs = sum(1 for job in jobs if job.status is JobStatus.FAILED)
    next_actions = _attention_recommendations(
        pending_review=len(pending_review),
        failed_jobs=failed_jobs,
        paused=paused,
    )
    governance = evaluate_governance(store)
    debug_log_path = (
        (cfg.project_root / ".claude" / "mm.debug.log")
        if cfg.project_root is not None
        else (cfg.db_path.parent / "mm.debug.log")
    )
    debug_log_present = debug_log_path.exists()
    retrieval_telemetry = _read_retrieval_telemetry(debug_log_path)
    proof = compute_proof_metrics(store)

    # Top and struggling skills
    top_skills = [s for s in skills if s.maturity is not Maturity.CANDIDATE and s.invocations >= 2][
        :3
    ]
    struggling = sorted(
        [s for s in skills if s.invocations >= 3 and s.score < 0.5],
        key=lambda s: s.score,
    )[:3]

    # -- JSON output --
    if as_json:
        data = {
            "database": str(cfg.db_path),
            "status": "paused" if paused else "active",
            "scope": cfg.scope.value,
            "pool_used": len(skills),
            "pool_max": cfg.max_skills,
            "episodes_total": episodes_total,
            "episodes_with_skills": len(eps_with_skills),
            "episodes_with_skills_pct": (len(eps_with_skills) / len(episodes) if episodes else 0.0),
            "reuse_rate": reuse_rate,
            "avg_reward": avg_reward,
            "maturity": {m.value: by_maturity[m] for m in Maturity},
            "new_7d": new_7d,
            "last_learned_at": last_learned_at.isoformat() if last_learned_at else None,
            "refined_skills": len(refined_skills),
            "total_refinements": total_refinements,
            "attention": {
                "need_refine": len(need_refine),
                "at_risk": len(at_risk),
                "counter_drift": len(inconsistent_counters),
                "pending_review": len(pending_review),
                "stale": len(stale),
                "unknown_rate": unknown_rate,
                "pending_jobs": pending_jobs,
                "failed_jobs": failed_jobs,
                "debug_log_present": debug_log_present,
                "retrieval_samples": (retrieval_telemetry or {}).get("samples", 0),
                "avg_retrieve_ms": (retrieval_telemetry or {}).get("avg_retrieve_ms", 0.0),
                "next_actions": next_actions,
            },
            "governance": {
                "demote": governance.demote_skill_ids,
                "refine": governance.refine_skill_ids,
                "review": governance.review_skill_ids,
            },
            "proof": {
                "confidence": proof.confidence.value,
                "comparable_tasks": proof.comparable_tasks,
                "assisted_tasks": proof.assisted_tasks,
                "unassisted_tasks": proof.unassisted_tasks,
                "assisted_success_rate": proof.assisted_success_rate,
                "unassisted_success_rate": proof.unassisted_success_rate,
                "outcome_lift": proof.outcome_lift,
                "token_reduction": proof.token_reduction,
                "token_samples": proof.token_samples,
                "unknown_outcomes": proof.unknown_outcomes,
            },
            "top_skills": [_skill_to_dict(s) for s in top_skills],
            "struggling_skills": [_skill_to_dict(s) for s in struggling],
        }
        typer.echo(json.dumps(data, indent=2))
        return

    # -- Rich output --

    # Section 1: Header
    status_line = "[yellow]PAUSED[/yellow]" if paused else "[green]active[/green]"
    header = (
        f"[bold]database[/bold]  {cfg.db_path}\n"
        f"[bold]status[/bold]    {status_line}"
        f"   [bold]scope[/bold] {cfg.scope.value}"
        f"   [bold]pool[/bold] {len(skills)}/{cfg.max_skills}"
    )
    console.print(Panel(header, title="muscle-memory"))

    console.print(Rule("Proof"))
    if proof.comparable_tasks < 10:
        console.print(
            f"  [yellow]insufficient evidence[/yellow]  ({proof.comparable_tasks} comparable tasks)"
        )
        console.print(
            "  [dim]Need at least 10 comparable measured tasks with assisted "
            "and unassisted examples.[/dim]"
        )
    else:
        lift = (
            f"{proof.outcome_lift:+.1%}"
            if proof.outcome_lift is not None
            else "not enough paired outcomes"
        )
        token_delta = (
            f"{proof.token_reduction:.1%}"
            if proof.token_reduction is not None
            else "not enough token samples"
        )
        console.print(f"  [bold]outcome lift[/bold]    {lift}")
        console.print(f"  [bold]token reduction[/bold] {token_delta}")
        console.print(
            f"  [bold]confidence[/bold]      {proof.confidence.value}"
            f"  ({proof.comparable_tasks} comparable tasks)"
        )
        if proof.unknown_outcomes:
            console.print(f"  [yellow]unknown outcomes[/yellow] {proof.unknown_outcomes}")

    # Empty store shortcut
    if not skills and not episodes:
        console.print(
            "\n[dim]No skills yet. Run [bold]mm bootstrap[/bold]"
            " to seed from session history.[/dim]"
        )
        return

    # Section 2: Value
    console.print(Rule("Value"))
    eps_note = " (last 1000)" if len(episodes) == 1000 else ""
    with_pct = f"{len(eps_with_skills) / len(episodes):.1%}" if episodes else "0.0%"
    console.print(
        f"  [bold]episodes[/bold]       {len(episodes)}{eps_note}"
        f"       [bold]with skills[/bold]   {len(eps_with_skills)} ({with_pct})"
    )
    if total_invocations:
        console.print(
            f"  [bold]reuse rate[/bold]     {reuse_rate:.1%}"
            f"        [bold]avg reward[/bold]    {avg_reward:+.2f}"
        )

    # Section 3: Learning
    console.print(Rule("Learning"))
    maturity_line = (
        f"{by_maturity[Maturity.PROVEN]} proven"
        f" · {by_maturity[Maturity.LIVE]} live"
        f" · {by_maturity[Maturity.CANDIDATE]} candidate"
    )
    console.print(f"  [bold]maturity[/bold]       {maturity_line}")
    last_learned_str = _relative_time(last_learned_at) if last_learned_at else "never"
    console.print(
        f"  [bold]new (7d)[/bold]       {new_7d} skills"
        f"      [bold]last learned[/bold]  {last_learned_str}"
    )
    if refined_skills:
        console.print(
            f"  [bold]refined[/bold]        {len(refined_skills)} skills"
            f" ({total_refinements} total refinements)"
        )

    # Section 4: Attention
    console.print(Rule("Attention"))
    attention_items = 0
    if need_refine:
        console.print(
            f"  [yellow]need refine[/yellow]    {len(need_refine)} skills"
            "  (≥5 uses, score ≤0.6, ≥2 failures)"
        )
        attention_items += 1
    if at_risk:
        console.print(
            f"  [red]at risk[/red]        {len(at_risk)} skills"
            "  (≥5 uses, score ≤0.2 — will be pruned)"
        )
        attention_items += 1
    if pending_review:
        console.print(
            f"  [yellow]pending review[/yellow] {len(pending_review)} candidates"
            "  (advanced: `mm review list` / `mm review approve`)"
        )
        attention_items += 1
    if inconsistent_counters:
        console.print(
            f"  [yellow]counter drift[/yellow] {len(inconsistent_counters)} skills"
            "  (success/failure counts exceed activations; run `mm maint rescore`)"
        )
        attention_items += 1
    if stale:
        console.print(
            f"  [yellow]stale[/yellow]          {len(stale)} skills  (invoked but unused >30d)"
        )
        attention_items += 1
    if unknown_rate > 0.4 and episodes:
        console.print(
            f"  [yellow]unknown rate[/yellow]   {unknown_rate:.1%}"
            f"  ({unknown_count}/{len(episodes)} episodes)"
        )
        attention_items += 1
    if pending_jobs:
        console.print(
            f"  [yellow]pending jobs[/yellow]   {pending_jobs} job(s) awaiting completion"
        )
        attention_items += 1
    if failed_jobs:
        console.print(
            f"  [red]failed jobs[/red]    {failed_jobs} job(s) need retry"
            "  (advanced: `mm jobs retry-failed`)"
        )
        attention_items += 1
    if paused:
        console.print(
            "  [yellow]paused[/yellow]        project is paused"
            "  (advanced: `mm maint resume` before dogfooding)"
        )
        attention_items += 1
    if governance.demote_skill_ids:
        console.print(
            f"  [red]eval demote[/red]    {len(governance.demote_skill_ids)} skill(s) show poor health"
        )
        attention_items += 1
    if governance.review_skill_ids:
        console.print(
            f"  [yellow]eval review[/yellow]   {len(governance.review_skill_ids)} skill(s) need review"
        )
        attention_items += 1
    if attention_items == 0:
        console.print("  [green]No issues detected.[/green]")
    elif next_actions:
        console.print("  [bold]next actions[/bold]")
        for action in next_actions:
            console.print(f"    - {action}")

    # Section 5: Top skills
    if top_skills:
        console.print(Rule("Top Skills"))
        top_table = Table(box=None, show_header=True, padding=(0, 1))
        top_table.add_column("id", style="dim", width=10)
        top_table.add_column("score", justify="right")
        top_table.add_column("uses", justify="right")
        top_table.add_column("maturity", width=12)
        top_table.add_column("activation", no_wrap=True)
        for s in top_skills:
            top_table.add_row(
                _short_id(s.id),
                f"{s.score:.2f}",
                f"{s.successes}/{s.invocations}",
                _format_maturity(s.maturity),
                s.activation[:40] + ("…" if len(s.activation) > 40 else ""),
            )
        console.print(top_table)

    # Section 6: Struggling skills
    if struggling:
        console.print(Rule("Struggling Skills"))
        str_table = Table(box=None, show_header=True, padding=(0, 1))
        str_table.add_column("id", style="dim", width=10)
        str_table.add_column("score", justify="right")
        str_table.add_column("uses", justify="right")
        str_table.add_column("maturity", width=12)
        str_table.add_column("activation", no_wrap=True)
        for s in struggling:
            str_table.add_row(
                _short_id(s.id),
                f"{s.score:.2f}",
                f"{s.successes}/{s.invocations}",
                _format_maturity(s.maturity),
                s.activation[:40] + ("…" if len(s.activation) > 40 else ""),
            )
        console.print(str_table)


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Inspect runtime health for database, jobs, and debug logging."""
    cfg = _load_config()
    db_exists = cfg.db_path.exists()
    paused = (cfg.db_path.parent / "mm.paused").exists()
    debug_log_path = (
        (cfg.project_root / ".claude" / "mm.debug.log")
        if cfg.project_root is not None
        else (cfg.db_path.parent / "mm.debug.log")
    )
    debug_log_exists = debug_log_path.exists()

    jobs: list[BackgroundJob] = []
    if db_exists:
        store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
        jobs = store.list_jobs(limit=None)

    counts = {status.value: 0 for status in JobStatus}
    for job in jobs:
        counts[job.status.value] += 1

    last_debug_event = _read_last_debug_event(debug_log_path)
    retrieval_telemetry = _read_retrieval_telemetry(debug_log_path)
    recent_retrieval_decisions = _read_recent_retrieval_decisions(debug_log_path)
    recommendations = _doctor_recommendations(debug_enabled=cfg.debug_enabled, paused=paused)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "database": str(cfg.db_path),
                    "db_exists": db_exists,
                    "paused": paused,
                    "debug_enabled": cfg.debug_enabled,
                    "debug_log_path": str(debug_log_path),
                    "debug_log_exists": debug_log_exists,
                    "job_counts": counts,
                    "last_debug_event": last_debug_event,
                    "retrieval_telemetry": retrieval_telemetry,
                    "recent_retrieval_decisions": recent_retrieval_decisions,
                    "recommendations": recommendations,
                },
                indent=2,
            )
        )
        return

    status_line = "[yellow]paused[/yellow]" if paused else "[green]active[/green]"
    console.print(
        Panel(
            f"[bold]database[/bold]  {cfg.db_path}\n[bold]status[/bold]    {status_line}",
            title="doctor",
        )
    )
    console.print(f"[bold]db exists[/bold]       {'yes' if db_exists else 'no'}")
    console.print(f"[bold]debug enabled[/bold]  {'yes' if cfg.debug_enabled else 'no'}")
    console.print(
        f"[bold]debug log[/bold]      {debug_log_path} ({'present' if debug_log_exists else 'missing'})"
    )
    console.print(
        f"[bold]jobs[/bold]           pending={counts['pending']} running={counts['running']} "
        f"failed={counts['failed']} succeeded={counts['succeeded']}"
    )
    if last_debug_event:
        console.print(
            f"[bold]last event[/bold]     {last_debug_event.get('component', '?')}::"
            f"{last_debug_event.get('event', '?')}"
        )
    if retrieval_telemetry:
        console.print(
            f"[bold]retrieval[/bold]      samples={retrieval_telemetry['samples']} "
            f"avg_retrieve_ms={retrieval_telemetry['avg_retrieve_ms']} "
            f"avg_total_ms={retrieval_telemetry['avg_total_ms']}"
        )
    if recent_retrieval_decisions:
        latest = recent_retrieval_decisions[0]
        console.print(f"[bold]latest decision[/bold] {latest['event']} — {latest['why']}")
    if recommendations:
        console.print(Rule("Recommendations"))
        for recommendation in recommendations:
            console.print(f"  - {recommendation}")
    if not db_exists:
        console.print("[dim]Database missing. Run [bold]mm init[/bold] first.[/dim]")


@app.command(hidden=True)
def refine(
    skill_id: str | None = typer.Argument(
        None, help="Skill id or prefix to refine. Omit with --auto to sweep."
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Refine every skill that meets auto-trigger criteria."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run refinement but do not persist changes."
    ),
    episodes: int = typer.Option(
        5,
        "--episodes",
        "-n",
        help="Max stored episodes to use as evidence per refinement.",
    ),
    rollback: bool = typer.Option(
        False, "--rollback", help="Restore the previous skill text (undoes one refinement)."
    ),
    job_id: str | None = typer.Option(None, "--job-id", hidden=True),
) -> None:
    """Non-parametric PPO refinement of one or all skills.

    Runs a three-stage process per skill:

      1. Extract a semantic gradient from recent trajectories (LLM call)
      2. Rewrite the skill text applying that gradient (LLM call)
      3. Verify via LLM-judge across stored trajectories; accept only
         if the revision demonstrably outperforms the original

    Examples:

      mm refine eab176c6             # refine one specific skill
      mm refine --auto               # sweep all skills meeting criteria
      mm refine abc123 --dry-run     # dry-run a specific skill
      mm refine abc123 --rollback    # undo the most recent refinement
    """
    from muscle_memory.llm import make_llm
    from muscle_memory.refine import (
        RefinementResult,
        refine_skill,
        should_auto_refine,
    )

    cfg = _load_config()
    store = _open_store(cfg)
    if job_id:
        try:
            store.update_job_status(job_id, status=JobStatus.RUNNING)
        except KeyError:
            pass

    if rollback:
        if not skill_id:
            console.print("[red]--rollback requires a skill id.[/red]")
            raise typer.Exit(1)
        skill = _resolve_skill(store, skill_id)
        if not skill.previous_text:
            console.print(
                f"[yellow]Skill {skill.id[:8]} has no previous text to rollback to.[/yellow]"
            )
            raise typer.Exit(1)
        restored_activation = skill.previous_text.get("activation", skill.activation)
        restored_execution = skill.previous_text.get("execution", skill.execution)
        restored_termination = skill.previous_text.get("termination", skill.termination)
        skill.activation = restored_activation
        skill.execution = restored_execution
        skill.termination = restored_termination
        skill.previous_text = None
        skill.refinement_count = max(0, skill.refinement_count - 1)
        store.update_skill(skill)
        console.print(f"[green]Rolled back {skill.id[:8]} to previous text.[/green]")
        return

    targets: list[Skill] = []
    if auto:
        for s in store.list_skills():
            if should_auto_refine(s):
                targets.append(s)
        if not targets:
            console.print("[green]No skills meet auto-refine criteria.[/green]")
            if job_id:
                store.update_job_status(job_id, status=JobStatus.SUCCEEDED)
            return
        console.print(
            f"Sweeping [bold]{len(targets)}[/bold] skill(s) matching "
            f"auto-refine criteria (failures ≥ 2, score ≤ 0.6, invocations ≥ 5)."
        )
    else:
        if not skill_id:
            console.print("[red]Need a skill id, or --auto to sweep.[/red]")
            raise typer.Exit(1)
        targets = [_resolve_skill(store, skill_id)]

    llm = make_llm(cfg)
    results: list[RefinementResult] = []

    for skill in targets:
        console.print(
            f"\n[cyan]Refining {skill.id[:8]}[/cyan] "
            f"({skill.successes}/{skill.invocations} successes, "
            f"score {skill.score:.2f})"
        )
        ep_list = store.find_episodes_for_skill(skill.id, limit=episodes)
        if not ep_list:
            console.print("  [yellow]No stored trajectories — skipping[/yellow]")
            continue

        result = refine_skill(
            skill,
            ep_list,
            llm,
            store=None if dry_run else store,
        )
        results.append(result)
        _print_refinement_result(result, dry_run=dry_run)

    accepted = sum(1 for r in results if r.accepted)
    rejected = len(results) - accepted
    console.print(
        f"\n[bold]Refinement complete.[/bold] "
        f"Accepted: [green]{accepted}[/green]   "
        f"Rejected: [yellow]{rejected}[/yellow]"
        + ("   [dim](dry-run — no changes written)[/dim]" if dry_run else "")
    )
    if job_id:
        store.update_job_status(job_id, status=JobStatus.SUCCEEDED)


def _print_refinement_result(result: Any, *, dry_run: bool) -> None:
    from rich.panel import Panel

    if not result.accepted:
        console.print(f"  [yellow]✗ rejected[/yellow]: {result.rejection_reason or 'unknown'}")
        if result.verdicts:
            console.print(
                f"    mean judge score: {result.mean_judge_score:+.2f} "
                f"across {len(result.verdicts)} verdicts"
            )
        return

    console.print(
        f"  [green]✓ accepted[/green] (mean judge {result.mean_judge_score:+.2f} "
        f"across {len(result.verdicts)} trajectories)"
    )
    if result.gradient:
        console.print(f"    [dim]root cause:[/dim] {result.gradient.root_cause}")
    if dry_run and result.revised_skill:
        body = (
            f"[bold]activation[/bold]\n{result.revised_skill.activation}\n\n"
            f"[bold]execution[/bold]\n{result.revised_skill.execution}\n\n"
            f"[bold]termination[/bold]\n{result.revised_skill.termination}"
        )
        console.print(Panel(body, title="proposed revision (dry-run)"))


@maint_app.command("rescore")
@app.command(hidden=True)
def rescore() -> None:
    """Re-run the outcome heuristic on every episode and re-credit skills.

    Use this after upgrading muscle-memory or tweaking the outcome
    heuristic — it retroactively applies the new rules to historical
    episodes so skill scores reflect current logic.
    """
    from muscle_memory.outcomes import infer_outcome

    cfg = _load_config()
    store = _open_store(cfg)

    # First, reset skill counters so rescoring is idempotent and can
    # repair stores whose invocation counts drifted in older versions.
    reset_n = 0
    skills_by_id: dict[str, Skill] = {}
    for skill in store.list_skills():
        skill.invocations = 0
        skill.successes = 0
        skill.failures = 0
        skill.recompute_score()
        skill.recompute_maturity()
        skills_by_id[skill.id] = skill
        reset_n += 1

    # Iterate episodes, re-infer outcomes, and rebuild per-skill counters
    # from the episode activation records.
    episodes = store.list_episodes(limit=None)
    outcome_counts = {"success": 0, "failure": 0, "unknown": 0}

    for ep in episodes:
        signal = infer_outcome(
            ep.trajectory,
            user_followup=ep.trajectory.user_followup,
            any_skills_activated=bool(ep.activated_skills),
        )
        ep.outcome = signal.outcome
        ep.reward = signal.reward
        outcome_counts[signal.outcome.value] += 1
        store.update_episode_outcome(ep.id, outcome=signal.outcome, reward=signal.reward)

        for skill_id in dict.fromkeys(ep.activated_skills):
            activated_skill = skills_by_id.get(skill_id)
            if activated_skill is None:
                continue
            activated_skill.invocations += 1
            if ep.outcome is Outcome.SUCCESS:
                activated_skill.successes += 1
            elif ep.outcome is Outcome.FAILURE:
                activated_skill.failures += 1

    for skill in skills_by_id.values():
        skill.recompute_score()
        skill.recompute_maturity()
        store.update_skill(skill)

    console.print(
        Panel.fit(
            f"reset {reset_n} skills\n"
            f"re-scored {len(episodes)} episodes\n"
            f"  success: {outcome_counts['success']}\n"
            f"  failure: {outcome_counts['failure']}\n"
            f"  unknown: {outcome_counts['unknown']}",
            title="rescore complete",
        )
    )


@maint_app.command("prune")
@app.command(hidden=True)
def prune(
    below: float = typer.Option(0.2, "--below", help="Score floor for pruning."),
    min_invocations: int = typer.Option(5, "--min-invocations"),
) -> None:
    """Remove demonstrably bad skills and enforce the capacity limit."""
    from muscle_memory.scorer import Scorer

    cfg = _load_config()
    store = _open_store(cfg)
    scorer = Scorer(store, max_skills=cfg.max_skills)
    # override the scorer's default by overwriting skills below the floor
    for skill in store.list_skills():
        if skill.invocations >= min_invocations and skill.score <= below:
            store.delete_skill(skill.id)

    # still run the capacity pass
    report = scorer.prune(min_invocations_before_prune=min_invocations)
    console.print(f"[green]Pruned {len(report.removed)} skills.[/green] {report.kept} remain.")


@maint_app.command("govern")
def govern(
    apply: bool = typer.Option(False, "--apply", help="Apply governance actions."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Recommend or apply eval-driven governance actions."""
    from muscle_memory.eval.evaluator import evaluate_governance

    cfg = _load_config()
    store = _open_store(cfg)
    report = evaluate_governance(store)
    payload = {
        "demote": report.demote_skill_ids,
        "refine": report.refine_skill_ids,
        "review": report.review_skill_ids,
    }

    if apply:
        for skill_id in report.demote_skill_ids:
            skill = store.get_skill(skill_id)
            if skill is None:
                continue
            skill.maturity = Maturity.CANDIDATE
            store.update_skill(skill)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(
        Panel.fit(
            f"demote: {len(report.demote_skill_ids)}\n"
            f"refine: {len(report.refine_skill_ids)}\n"
            f"review: {len(report.review_skill_ids)}\n"
            + ("\n[green]Applied demotions.[/green]" if apply else "\n[dim]Dry run only.[/dim]"),
            title="governance",
        )
    )


@share_app.command("export")
@app.command(hidden=True)
def export(
    output: Path = typer.Argument(Path("skills.json"), help="Output path."),
) -> None:
    """Export all skills to a JSON file."""
    cfg = _load_config()
    store = _open_store(cfg)
    skills = store.list_skills()
    payload = [_skill_to_dict(s) for s in skills]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Exported {len(skills)} skills to {output}[/green]")


@share_app.command("import")
@app.command("import", hidden=True)
def import_cmd(
    input: Path = typer.Argument(..., help="Input JSON file."),
) -> None:
    """Import skills from a JSON file (e.g., shared by a teammate)."""
    from muscle_memory.embeddings import make_embedder

    cfg = _load_config()
    store = _open_store(cfg)
    embedder = make_embedder(cfg)

    items = json.loads(input.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        console.print("[red]Input must be a JSON list of skills.[/red]")
        raise typer.Exit(1)

    added = 0
    for item in items:
        try:
            skill = Skill(**{k: v for k, v in item.items() if k != "id"})
            embedding = embedder.embed_one(skill.activation)
            store.add_skill(skill, embedding=embedding)
            added += 1
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Skipped invalid skill: {e}[/yellow]")
    console.print(f"[green]Imported {added} skills.[/green]")


@app.command()
def learn(
    transcript: Path | None = typer.Option(
        None,
        "--transcript",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Transcript file to learn from. Omit to scan recent Claude Code history.",
    ),
    format: str = typer.Option("claude-jsonl", "--format", help="Transcript format."),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Original user prompt for transcript formats that do not preserve it.",
    ),
    extract: bool = typer.Option(
        True, "--extract/--no-extract", help="Extract skills after ingesting a transcript."
    ),
    days: int = typer.Option(30, "--days", "-d", help="Look back N days when scanning history."),
    max_sessions: int = typer.Option(200, "--max-sessions"),
    project_only: bool = typer.Option(
        True,
        "--project-only/--all-projects",
        help="Only consider sessions from the current project when scanning history.",
    ),
) -> None:
    """Learn skills from recent history or an explicit transcript."""
    if transcript is not None:
        ingest_transcript_cmd(
            transcript=transcript,
            format=format,
            prompt=prompt,
            extract=extract,
        )
        return

    bootstrap(days=days, max_sessions=max_sessions, project_only=project_only)


@app.command(hidden=True)
def bootstrap(
    days: int = typer.Option(30, "--days", "-d", help="Look back N days."),
    max_sessions: int = typer.Option(200, "--max-sessions"),
    project_only: bool = typer.Option(
        True,
        "--project-only/--all-projects",
        help="Only consider sessions from the current project.",
    ),
) -> None:
    """Seed the skill store from existing Claude Code session history."""
    from muscle_memory.bootstrap import bootstrap as run_bootstrap
    from muscle_memory.llm import make_llm

    cfg = _load_config()
    cfg.ensure_db_dir()
    store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)

    console.print("[dim]Loading embedder (first run may download the model)…[/dim]")
    embedder = make_embedder(cfg)
    console.print("[dim]Loading LLM…[/dim]")
    llm = make_llm(cfg)

    with console.status("Processing session history…"):
        report = run_bootstrap(
            config=cfg,
            store=store,
            embedder=embedder,
            llm=llm,
            days=days,
            project_only=project_only,
            max_sessions=max_sessions,
        )

    title = "bootstrap aborted" if report.aborted_reason else "bootstrap complete"
    console.print(
        Panel.fit(
            f"sessions considered: {report.sessions_considered}\n"
            f"sessions parsed:    {report.sessions_parsed}\n"
            f"episodes added:     {report.episodes_added}\n"
            f"skills extracted:   {report.skills_extracted}\n"
            + (f"errors:             {len(report.errors)}" if report.errors else ""),
            title=title,
        )
    )
    if report.errors:
        for err in report.errors[:5]:
            console.print(f"[yellow]  {err}[/yellow]")
    if report.aborted_reason:
        console.print(
            f"\n[red]aborted:[/red] {report.aborted_reason}\n"
            "[dim]fix the underlying issue above (login/auth, Claude CLI compatibility, "
            "model name, or provider limits) "
            "and re-run `mm bootstrap`.[/dim]"
        )
        raise typer.Exit(1)


@ingest_app.command("transcript")
def ingest_transcript_cmd(
    transcript: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    format: str = typer.Option("claude-jsonl", "--format", help="Transcript format."),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Original user prompt for transcript formats that do not preserve it (for example codex-jsonl).",
    ),
    extract: bool = typer.Option(
        True, "--extract/--no-extract", help="Extract skills after ingesting."
    ),
) -> None:
    """Ingest a transcript from a supported harness format."""
    from muscle_memory.ingest import ingest_transcript_file

    cfg = _load_config()
    cfg.ensure_db_dir()
    store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
    episode, added = ingest_transcript_file(
        transcript,
        format,
        config=cfg,
        store=store,
        extract=extract,
        prompt_override=prompt,
    )
    console.print(
        f"[green]Ingested episode {episode.id[:8]}[/green] from {transcript}"
        f" ([dim]{added} skills added[/dim])"
    )


@ingest_app.command("episode")
def ingest_episode_cmd(
    episode_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    extract: bool = typer.Option(
        True, "--extract/--no-extract", help="Extract skills after ingesting."
    ),
) -> None:
    """Ingest a normalized episode JSON file."""
    from muscle_memory.ingest import ingest_episode_file

    cfg = _load_config()
    cfg.ensure_db_dir()
    store = Store(cfg.db_path, embedding_dims=cfg.embedding_dims)
    episode, added = ingest_episode_file(episode_file, config=cfg, store=store, extract=extract)
    console.print(
        f"[green]Ingested episode {episode.id[:8]}[/green] from {episode_file}"
        f" ([dim]{added} skills added[/dim])"
    )


@app.command("extract-episode", hidden=True)
def extract_episode_cmd(
    episode_id: str,
    job_id: str | None = typer.Option(None, "--job-id", hidden=True),
) -> None:
    """Extract skills from a single episode (used by the async pipeline)."""
    from muscle_memory.dedup import add_skill_with_dedup
    from muscle_memory.embeddings import make_embedder
    from muscle_memory.extractor import Extractor
    from muscle_memory.llm import make_llm

    cfg = _load_config()
    store = _open_store(cfg)
    if job_id:
        try:
            store.update_job_status(job_id, status=JobStatus.RUNNING)
        except KeyError:
            pass

    try:
        episode = store.get_episode(episode_id)
        if episode is None:
            console.print(f"[red]Episode {episode_id} not found[/red]")
            raise typer.Exit(1)

        llm = make_llm(cfg)
        embedder = make_embedder(cfg)
        ex = Extractor(llm, cfg)
        skills = ex.extract(episode)

        added = 0
        deduped = 0
        for skill in skills:
            was_added, _existing = add_skill_with_dedup(store, embedder, skill)
            if was_added:
                added += 1
            else:
                deduped += 1
        console.print(
            f"[green]Extracted {added} new skills from episode {episode_id}"
            f"[/green] ([dim]{deduped} deduped[/dim])"
        )
        if job_id:
            store.update_job_status(job_id, status=JobStatus.SUCCEEDED)
    except typer.Exit as exc:
        if job_id:
            store.update_job_status(job_id, status=JobStatus.FAILED, error=f"exit {exc.exit_code}")
        raise
    except Exception as exc:
        if job_id:
            store.update_job_status(job_id, status=JobStatus.FAILED, error=str(exc))
        raise


@maint_app.command("dedup")
@app.command(hidden=True)
def dedup(
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would be consolidated."),
    threshold: float = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Embedding distance threshold (0.0–2.0). Smaller = stricter.",
    ),
) -> None:
    """Find and collapse near-duplicate skills.

    Clusters skills by embedding similarity, then merges each cluster
    into its best-scoring member — summing successes/failures/invocations
    so history is preserved, and deleting the losers.
    """
    from muscle_memory.dedup import (
        DEDUP_DISTANCE_THRESHOLD,
        consolidate_group,
        find_near_duplicate_groups,
    )
    from muscle_memory.embeddings import make_embedder

    cfg = _load_config()
    store = _open_store(cfg)
    embedder = make_embedder(cfg)

    effective_threshold = threshold if threshold is not None else DEDUP_DISTANCE_THRESHOLD
    groups = find_near_duplicate_groups(store, embedder, effective_threshold)
    if not groups:
        console.print("[green]No near-duplicate skills found.[/green]")
        return

    total_losers = sum(len(g) - 1 for g in groups)
    console.print(
        f"Found [bold]{len(groups)}[/bold] duplicate group(s), "
        f"[bold]{total_losers}[/bold] skills would be collapsed.\n"
    )

    for i, group in enumerate(groups, start=1):
        keeper = group[0]
        console.print(
            f"[cyan]Group {i}[/cyan] — keep [bold]{keeper.id[:8]}[/bold] "
            f"({keeper.successes}/{keeper.invocations}, score {keeper.score:.2f})"
        )
        console.print(f"  [dim]{keeper.activation[:90]}…[/dim]")
        for loser in group[1:]:
            console.print(
                f"  [red]drop[/red] {loser.id[:8]} "
                f"({loser.successes}/{loser.invocations}, score {loser.score:.2f})"
            )
        console.print()

    if dry_run:
        console.print("[yellow]--dry-run: no changes made.[/yellow]")
        return

    consolidated = 0
    for group in groups:
        consolidate_group(store, group)
        consolidated += len(group) - 1

    console.print(
        f"[green]Collapsed {consolidated} duplicate skills.[/green] {store.count_skills()} remain."
    )


# ----------------------------------------------------------------------
# hook subcommands
# ----------------------------------------------------------------------


@hook_app.command("user-prompt")
def hook_user_prompt() -> None:
    """Handle a UserPromptSubmit event. Reads JSON on stdin."""
    from muscle_memory.hooks.user_prompt import main as up_main

    raise typer.Exit(up_main())


@hook_app.command("stop")
def hook_stop() -> None:
    """Handle a Stop event. Reads JSON on stdin."""
    from muscle_memory.hooks.stop import main as stop_main

    raise typer.Exit(stop_main())


# ----------------------------------------------------------------------
# eval subcommands
# ----------------------------------------------------------------------


@eval_app.command("label")
def eval_label(
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Label skill credit assignments: did the skill actually help?"""
    from muscle_memory.eval.labeler import label_credits_interactive

    cfg = _load_config()
    store = _open_store(cfg)
    label_credits_interactive(store, limit=limit)


@eval_app.command("build")
def eval_build(
    output: str = typer.Option(None, "--output", "-o", help="Output path for benchmark JSON."),
) -> None:
    """Score all activations and export a frozen benchmark."""
    from pathlib import Path as _Path

    from muscle_memory.eval.benchmark import build_benchmark

    cfg = _load_config()
    store = _open_store(cfg)
    out = _Path(output) if output else None
    entries, path = build_benchmark(store, embedder=make_embedder(cfg), output_path=out)
    console.print(f"[green]Built benchmark:[/green] {len(entries)} entries -> {path}")


@eval_app.command("run")
def eval_run(
    benchmark: str = typer.Option(None, "--benchmark", "-b", help="Path to benchmark JSON."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Re-score against a frozen benchmark and show diffs."""
    from pathlib import Path as _Path

    from muscle_memory.eval.benchmark import run_benchmark

    cfg = _load_config()
    store = _open_store(cfg)

    path = _Path(benchmark) if benchmark else cfg.db_path.parent / "benchmark.json"

    if not path.exists():
        # soft_wrap=True keeps long paths intact; Rich otherwise breaks them
        # on dashes/dots which corrupts assertions that look for `path.name`.
        console.print(
            f"[red]No benchmark at {path}.[/red] Run [bold]mm eval build[/bold] first.",
            soft_wrap=True,
        )
        raise typer.Exit(1)

    result = run_benchmark(store, path, embedder=make_embedder(cfg))

    if as_json:
        payload = asdict(result)
        payload["benchmark_path"] = str(path.resolve())
        payload["benchmark_sha256"] = _file_sha256(path)
        payload["db_path"] = str(cfg.db_path.resolve())
        payload["db_sha256"] = _file_sha256(cfg.db_path)
        if cfg.project_root is not None:
            payload["repo_root"] = str(cfg.project_root.resolve())
            payload["repo_head"] = _current_repo_head(cfg.project_root)
            payload["source_tree_sha256"] = _current_source_tree_sha256(cfg.project_root)
            worktree_clean, worktree_state = _current_worktree_state(cfg.project_root)
            payload["worktree_clean"] = worktree_clean
            payload["worktree_state"] = worktree_state
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(0 if result.thresholds_passed else 1)

    console.print(f"[bold]Benchmark Run[/bold] ({result.total} entries)\n")
    console.print(
        f"  [bold]Relevance:[/bold] {result.avg_relevance:.2f}"
        f"  (baseline {result.baseline_avg_relevance:.2f},"
        f"  delta {result.avg_relevance - result.baseline_avg_relevance:+.2f})"
    )
    console.print(
        f"  [bold]Adherence:[/bold] {result.avg_adherence:.2f}"
        f"  (baseline {result.baseline_avg_adherence:.2f},"
        f"  delta {result.avg_adherence - result.baseline_avg_adherence:+.2f})"
    )

    if result.improved:
        console.print(f"\n  [green]Improved ({len(result.improved)}):[/green]")
        for d in result.improved[:5]:
            console.print(f"    {d['skill_id']}  {d['activation']}")
    if result.degraded:
        console.print(f"\n  [red]Degraded ({len(result.degraded)}):[/red]")
        for d in result.degraded[:5]:
            console.print(f"    {d['skill_id']}  {d['activation']}")
    if not result.improved and not result.degraded:
        console.print("\n  [dim]No significant changes.[/dim]")
    console.print(
        f"\n  [bold]Release gate:[/bold] "
        f"{'[green]passed[/green]' if result.thresholds_passed else '[red]failed[/red]'}"
    )
    if result.failed_thresholds:
        console.print(f"  [red]Failed thresholds:[/red] {', '.join(result.failed_thresholds)}")

    raise typer.Exit(0 if result.thresholds_passed else 1)


@eval_app.command("report")
def eval_report() -> None:
    """Playbook health dashboard."""
    from muscle_memory.eval.evaluator import (
        evaluate_health,
        evaluate_impact,
        render_health_report,
        render_impact_eval,
    )

    cfg = _load_config()
    store = _open_store(cfg)

    skills = store.list_skills()
    counts = {m: 0 for m in Maturity}
    for skill in skills:
        counts[skill.maturity] += 1
    console.print(
        f"[bold]Pool:[/bold] {counts[Maturity.CANDIDATE]} candidate"
        f" · {counts[Maturity.LIVE]} live"
        f" · {counts[Maturity.PROVEN]} proven"
    )
    console.print()

    # Health
    health = evaluate_health(store)
    render_health_report(health)

    console.print()

    # Impact
    impact = evaluate_impact(store)
    render_impact_eval(impact)


# ----------------------------------------------------------------------
# simulate — synthetic dogfooding
# ----------------------------------------------------------------------


def _resolve_sim_db_path(db: Path | None, *, inplace: bool, cfg: Config) -> Path:
    """Decide which DB the simulator writes to.

    Priority: explicit --db > --inplace (project DB) > synthetic default.
    The synthetic default is ~/.claude/mm.sim.db — deliberately disjoint
    from any project's real skill pool.
    """
    from muscle_memory.simulate import default_sim_db_path

    if db is not None:
        return db
    if inplace:
        return cfg.db_path
    return default_sim_db_path()


def _open_sim_store(db_path: Path) -> Store:
    """Open-or-create the synthetic DB. Unlike `_open_store`, absence is fine."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # 16 dims matches DeterministicEmbedder and is cheap; embeddings are
    # not produced by the simulator anyway, but the store needs a fixed dim.
    return Store(db_path, embedding_dims=16)


@simulate_app.command("scenarios")
def simulate_scenarios(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List the built-in demo scenarios (and the skill ids they target)."""
    from muscle_memory.simulate_fixtures import demo_scenarios, demo_skills

    scenarios = demo_scenarios()
    skills = {s.id: s for s in demo_skills()}

    if as_json:
        payload = [
            {
                "name": sc.name,
                "prompt": sc.prompt,
                "activated_skills": sc.activated_skills,
                "success_rate": sc.success_rate,
                "n": sc.n,
                "tags": sc.tags,
            }
            for sc in scenarios
        ]
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title=f"demo scenarios ({len(scenarios)})")
    table.add_column("name", style="cyan")
    table.add_column("skill", style="dim")
    table.add_column("success", justify="right")
    table.add_column("n", justify="right")
    table.add_column("activation")
    for sc in scenarios:
        activation = " / ".join(
            skills[sid].activation if sid in skills else "(unknown skill)"
            for sid in sc.activated_skills
        )
        table.add_row(
            sc.name,
            ", ".join(sc.activated_skills),
            f"{sc.success_rate:.0%}",
            str(sc.n),
            activation,
        )
    console.print(table)


@simulate_app.command("seed")
def simulate_seed(
    db: Path | None = typer.Option(
        None, "--db", help="DB path override; defaults to ~/.claude/mm.sim.db."
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Write into the project DB. Off by default to protect real skills.",
    ),
) -> None:
    """Seed the synthetic DB with the built-in CANDIDATE skill fixtures."""
    from muscle_memory.simulate import Simulator
    from muscle_memory.simulate_fixtures import demo_skills

    cfg = _load_config()
    db_path = _resolve_sim_db_path(db, inplace=inplace, cfg=cfg)
    store = _open_sim_store(db_path)
    sim = Simulator(store)

    seeded = sim.seed(demo_skills())
    console.print(f"[green]Seeded {len(seeded)} skills[/green] into [bold]{db_path}[/bold].")
    for skill_id in seeded:
        console.print(f"  [dim]{skill_id}[/dim]")


@simulate_app.command("run")
def simulate_run(
    db: Path | None = typer.Option(
        None, "--db", help="DB path override; defaults to ~/.claude/mm.sim.db."
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Write into the project DB. Off by default to protect real skills.",
    ),
    seed_value: int | None = typer.Option(
        None, "--seed", help="RNG seed for reproducible success/failure mixes."
    ),
    prune: bool | None = typer.Option(
        None,
        "--prune/--no-prune",
        help=(
            "Run the pruner after scenarios finish. "
            "Defaults to True for the isolated sim DB, False with --inplace "
            "(opt in explicitly to prune real project skills)."
        ),
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Delete the simulate DB before running. Ignored with --inplace.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Run the built-in demo scenarios end-to-end.

    Pipeline per scenario: seed skills if missing → generate N episodes
    with the declared success rate → credit each episode via the Scorer
    → optionally prune. After a run, inspect end-state with `mm list`
    against the same `--db`.
    """
    import random as _random

    from muscle_memory.simulate import Simulator, default_sim_db_path
    from muscle_memory.simulate_fixtures import demo_scenarios, demo_skills

    cfg = _load_config()
    db_path = _resolve_sim_db_path(db, inplace=inplace, cfg=cfg)

    # Prune safety: the pruner will delete real skills (score <=0.2 at
    # >=5 invocations). Auto-prune is only safe on the canonical sim DB —
    # --inplace OR --db <project/.claude/mm.db> both target real skills
    # and must require explicit --prune.
    sim_default = default_sim_db_path()
    is_sim_db = db_path.resolve() == sim_default.resolve()
    effective_prune = is_sim_db if prune is None else prune
    if effective_prune and not is_sim_db:
        # stderr via typer.echo (not Rich console) so --json stdout stays pure.
        typer.echo(
            "Warning: --prune against a non-sim DB will remove any "
            "skill whose score <=0.2 at >=5 invocations.",
            err=True,
        )

    if fresh and not inplace and db_path.exists():
        db_path.unlink()
        # sqlite WAL sidecars
        for suffix in ("-wal", "-shm"):
            side = db_path.with_name(db_path.name + suffix)
            if side.exists():
                side.unlink()

    store = _open_sim_store(db_path)
    rng = _random.Random(seed_value) if seed_value is not None else _random.Random()
    sim = Simulator(store, rng=rng)

    sim.seed(demo_skills())
    report = sim.run(demo_scenarios(), prune=effective_prune)

    if as_json:
        payload = {
            "db_path": str(db_path),
            "total_episodes": report.total_episodes,
            "scenarios": [
                {
                    "name": r.name,
                    "successes": r.successes,
                    "failures": r.failures,
                    "episodes_written": r.episodes_written,
                }
                for r in report.scenarios
            ],
            "pruned": (report.prune_report.removed if report.prune_report else []),
            "kept": (report.prune_report.kept if report.prune_report else None),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title=f"simulation results — {db_path}")
    table.add_column("scenario", style="cyan")
    table.add_column("successes", justify="right", style="green")
    table.add_column("failures", justify="right", style="red")
    table.add_column("episodes", justify="right")
    for r in report.scenarios:
        table.add_row(
            r.name,
            str(r.successes),
            str(r.failures),
            str(r.episodes_written),
        )
    console.print(table)
    console.print(f"\n[bold]{report.summary_line()}[/bold]")
    console.print(
        f"\nInspect with: [bold]mm list --limit 50[/bold] "
        f"(set MM_DB_PATH={db_path} first, or use `sqlite3 {db_path}`)."
    )


# ----------------------------------------------------------------------
# utility
# ----------------------------------------------------------------------


def _resolve_skill(store: Store, id_or_prefix: str) -> Skill:
    skill = store.get_skill(id_or_prefix)
    if skill is not None:
        return skill
    # prefix match
    matches = [s for s in store.list_skills() if s.id.startswith(id_or_prefix)]
    if not matches:
        console.print(f"[red]No skill matching {id_or_prefix!r}[/red]")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[red]Ambiguous: {len(matches)} skills start with {id_or_prefix!r}[/red]")
        raise typer.Exit(1)
    return matches[0]


def _resolve_job(store: Store, id_or_prefix: str) -> BackgroundJob:
    job = store.get_job(id_or_prefix)
    if job is not None:
        return job
    matches = [j for j in store.list_jobs(limit=None) if j.id.startswith(id_or_prefix)]
    if not matches:
        console.print(f"[red]No job matching {id_or_prefix!r}[/red]")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[red]Ambiguous: {len(matches)} jobs start with {id_or_prefix!r}[/red]")
        raise typer.Exit(1)
    return matches[0]


def _candidate_review_metadata(skill: Skill) -> dict[str, Any]:
    evidence = len(dict.fromkeys(skill.source_episode_ids))
    auto_ready = evidence >= 2 and skill.successes >= 2 and skill.score >= 0.6
    if auto_ready:
        reason = "ready to promote"
    elif evidence < 2:
        reason = "needs more evidence"
    elif skill.successes < 2:
        reason = "needs more successful runs"
    elif skill.score < 0.6:
        reason = "performance below live threshold"
    else:
        reason = "awaiting review"
    return {
        "source_evidence": evidence,
        "auto_promote_ready": auto_ready,
        "review_reason": reason,
    }


def _attention_recommendations(*, pending_review: int, failed_jobs: int, paused: bool) -> list[str]:
    recommendations: list[str] = []
    if pending_review:
        recommendations.append("Run advanced `mm review list` to inspect quarantined candidates.")
    if failed_jobs:
        recommendations.append(
            "Run advanced `mm jobs retry-failed` to retry failed background work."
        )
    if paused:
        recommendations.append(
            "Run advanced `mm maint resume` before dogfooding if the project is paused."
        )
    return recommendations


def _doctor_recommendations(*, debug_enabled: bool, paused: bool) -> list[str]:
    recommendations: list[str] = []
    if not debug_enabled:
        recommendations.append(
            "Enable MM_DEBUG=1 while validating Claude Code retrieval decisions."
        )
    if paused:
        recommendations.append(
            "Run advanced `mm maint resume` before dogfooding if the project is paused."
        )
    return recommendations


def _read_last_debug_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _read_retrieval_telemetry(path: Path) -> dict[str, float | int] | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    samples = 0
    total_retrieve = 0.0
    total_embed = 0.0
    total_search = 0.0
    total_rerank = 0.0
    total_total = 0.0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("component") != "user_prompt":
            continue
        if data.get("event") not in {"hits_returned", "no_hits"}:
            continue
        retrieve_ms = data.get("retrieve_ms")
        if retrieve_ms is None:
            continue
        samples += 1
        total_retrieve += float(retrieve_ms)
        total_embed += float(data.get("embed_ms") or 0.0)
        total_search += float(data.get("search_ms") or 0.0)
        total_rerank += float(data.get("rerank_ms") or 0.0)
        total_total += float(data.get("total_ms") or 0.0)

    if samples == 0:
        return None
    return {
        "samples": samples,
        "avg_retrieve_ms": round(total_retrieve / samples, 3),
        "avg_embed_ms": round(total_embed / samples, 3),
        "avg_search_ms": round(total_search / samples, 3),
        "avg_rerank_ms": round(total_rerank / samples, 3),
        "avg_total_ms": round(total_total / samples, 3),
    }


def _read_recent_retrieval_decisions(path: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    decisions: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("component") != "user_prompt":
            continue
        event = str(data.get("event") or "")
        if event not in {"hits_returned", "no_hits", "shell_escape_skip", "no_db", "paused"}:
            continue
        why = "retrieved matching trusted skills"
        if event == "no_hits":
            reject_reason = str(data.get("reject_reason") or "")
            if reject_reason == "weak_match_without_lexical_support":
                why = "weak semantic hit without lexical corroboration"
            elif reject_reason == "distance_below_similarity_floor":
                why = "semantic match fell below the configured similarity floor"
            elif reject_reason == "distance_above_weak_match_window":
                why = "semantic match fell outside the weak-match window"
            elif reject_reason == "no lexical overlap with trusted skills" or data.get(
                "lexical_prefilter_skipped"
            ):
                why = "no lexical overlap with trusted skills"
            else:
                why = "no trusted skills passed retrieval filters"
        elif event == "shell_escape_skip":
            why = "prompt looked like a direct shell/slash command"
        elif event == "no_db":
            why = "database not initialized"
        elif event == "paused":
            why = "muscle-memory is paused"
        decisions.append(
            {
                "timestamp": data.get("timestamp"),
                "session_id": data.get("session_id"),
                "event": event,
                "prompt_excerpt": data.get("prompt_excerpt"),
                "why": why,
                "hit_count": data.get("hit_count", 0),
            }
        )
        if len(decisions) >= limit:
            break
    return decisions


def _skill_to_dict(s: Skill) -> dict[str, Any]:
    return {
        "id": s.id,
        "activation": s.activation,
        "execution": s.execution,
        "termination": s.termination,
        "tool_hints": s.tool_hints,
        "tags": s.tags,
        "scope": s.scope.value,
        "score": s.score,
        "invocations": s.invocations,
        "successes": s.successes,
        "failures": s.failures,
        "maturity": s.maturity.value,
        "source_episode_ids": s.source_episode_ids,
        "created_at": s.created_at.isoformat(),
        "last_used_at": s.last_used_at.isoformat() if s.last_used_at else None,
    }


def _job_to_dict(job: BackgroundJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind.value,
        "status": job.status.value,
        "payload": job.payload,
        "attempts": job.attempts,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


if __name__ == "__main__":
    app()
