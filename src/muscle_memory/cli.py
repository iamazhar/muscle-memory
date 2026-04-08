"""`mm` — command-line interface for muscle-memory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from muscle_memory import __version__
from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.models import Maturity, Outcome, Scope, Skill

console = Console()
app = typer.Typer(
    name="mm",
    help="muscle-memory: procedural memory for coding agents.",
    no_args_is_help=True,
    add_completion=False,
)

hook_app = typer.Typer(help="Claude Code hook handlers (not for direct use).")
app.add_typer(hook_app, name="hook")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _load_config(scope: Scope | None = None) -> Config:
    return Config.load(scope=scope)


def _open_store(cfg: Config) -> Store:
    if not cfg.db_path.exists():
        console.print(
            f"[red]No muscle-memory database at {cfg.db_path}.[/red]\n"
            "Run [bold]mm init[/bold] inside a project first."
        )
        raise typer.Exit(2)
    return Store(cfg.db_path, embedding_dims=cfg.embedding_dims)


def _format_maturity(m: Maturity) -> str:
    colors = {
        Maturity.CANDIDATE: "yellow",
        Maturity.ESTABLISHED: "cyan",
        Maturity.PROVEN: "green",
    }
    return f"[{colors[m]}]{m.value}[/{colors[m]}]"


def _short_id(s: str, n: int = 8) -> str:
    return s[:n]


# ----------------------------------------------------------------------
# top-level commands
# ----------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"muscle-memory {__version__}")


@app.command()
def init(
    scope: Scope = typer.Option(
        Scope.PROJECT,
        "--scope",
        "-s",
        help="project (default) or global",
    ),
) -> None:
    """Set up muscle-memory for the current project.

    Creates `.claude/mm.db` and wires `UserPromptSubmit` + `Stop` hooks
    into `.claude/settings.json`.
    """
    from muscle_memory.hooks.install import install as do_install

    try:
        report = do_install(scope=scope)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    console.print(
        Panel.fit(
            f"[green]muscle-memory initialized[/green]\n\n"
            f"DB: [bold]{report.db_path}[/bold]\n"
            f"Settings: [bold]{report.settings_path}[/bold]\n"
            f"Installed hooks: {', '.join(report.installed_events) or '(already present)'}\n"
            f"Already present: {', '.join(report.already_present) or '—'}",
            title="init complete",
        )
    )
    console.print(
        "\nNext: use Claude Code as usual. Optionally seed with "
        "[bold]mm bootstrap[/bold].",
    )


@app.command("list")
def list_skills(
    maturity: Optional[Maturity] = typer.Option(None, "--maturity", "-m"),
    limit: int = typer.Option(50, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List skills in the current project's database."""
    cfg = _load_config()
    store = _open_store(cfg)
    skills = store.list_skills(maturity=maturity, limit=limit)

    if as_json:
        typer.echo(json.dumps([_skill_to_dict(s) for s in skills], indent=2))
        return

    if not skills:
        console.print("[dim]No skills yet. Try [bold]mm bootstrap[/bold] to seed from history.[/dim]")
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
            f"{s.score:.2f}",
            f"{s.successes}/{s.invocations}",
            s.activation[:80] + ("…" if len(s.activation) > 80 else ""),
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
                f"[dim]score:[/dim] {skill.score:.2f}   "
                f"[dim]uses:[/dim] {skill.successes}/{skill.invocations}\n"
                f"[dim]tags:[/dim] {', '.join(skill.tags) or '—'}\n"
                f"[dim]tool hints:[/dim] {', '.join(skill.tool_hints) or '—'}"
            ),
            title=f"skill {_short_id(skill.id)}",
        )
    )


@app.command()
def stats() -> None:
    """Summarize the skill store."""
    cfg = _load_config()
    store = _open_store(cfg)
    skills = store.list_skills()
    episodes = store.list_episodes(limit=1000)

    by_maturity: dict[Maturity, int] = {}
    for m in Maturity:
        by_maturity[m] = 0
    for s in skills:
        by_maturity[s.maturity] += 1

    total_invocations = sum(s.invocations for s in skills)
    total_successes = sum(s.successes for s in skills)
    reuse_rate = (total_successes / total_invocations) if total_invocations else 0.0

    by_outcome: dict[Outcome, int] = {o: 0 for o in Outcome}
    for ep in episodes:
        by_outcome[ep.outcome] += 1

    panel_text = (
        f"[bold]database[/bold] {cfg.db_path}\n"
        f"[bold]project root[/bold] {cfg.project_root or '(global)'}\n\n"
        f"[bold]skills[/bold] {len(skills)}"
        f"  (candidate: {by_maturity[Maturity.CANDIDATE]},"
        f"  established: {by_maturity[Maturity.ESTABLISHED]},"
        f"  proven: {by_maturity[Maturity.PROVEN]})\n"
        f"[bold]invocations[/bold] {total_invocations}"
        f"  [bold]reuse rate[/bold] {reuse_rate:.1%}\n\n"
        f"[bold]episodes[/bold] {len(episodes)}"
        f"  (success: {by_outcome[Outcome.SUCCESS]},"
        f"  failure: {by_outcome[Outcome.FAILURE]},"
        f"  unknown: {by_outcome[Outcome.UNKNOWN]})"
    )
    console.print(Panel(panel_text, title="muscle-memory stats"))


@app.command()
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
    console.print(
        f"[green]Pruned {len(report.removed)} skills.[/green] "
        f"{report.kept} remain."
    )


@app.command()
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


@app.command("import")
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
def bootstrap(
    days: int = typer.Option(30, "--days", "-d", help="Look back N days."),
    max_sessions: int = typer.Option(200, "--max-sessions"),
    project_only: bool = typer.Option(
        True, "--project-only/--all-projects",
        help="Only consider sessions from the current project.",
    ),
) -> None:
    """Seed the skill store from existing Claude Code session history."""
    from muscle_memory.bootstrap import bootstrap as run_bootstrap
    from muscle_memory.embeddings import make_embedder
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

    console.print(
        Panel.fit(
            f"sessions considered: {report.sessions_considered}\n"
            f"sessions parsed:    {report.sessions_parsed}\n"
            f"episodes added:     {report.episodes_added}\n"
            f"skills extracted:   {report.skills_extracted}\n"
            + (f"errors:             {len(report.errors)}" if report.errors else ""),
            title="bootstrap complete",
        )
    )
    if report.errors:
        for err in report.errors[:5]:
            console.print(f"[yellow]  {err}[/yellow]")


@app.command("extract-episode", hidden=True)
def extract_episode_cmd(episode_id: str) -> None:
    """Extract skills from a single episode (used by the async pipeline)."""
    from muscle_memory.embeddings import make_embedder
    from muscle_memory.extractor import Extractor
    from muscle_memory.llm import make_llm

    cfg = _load_config()
    store = _open_store(cfg)

    episode = store.get_episode(episode_id)
    if episode is None:
        console.print(f"[red]Episode {episode_id} not found[/red]")
        raise typer.Exit(1)

    llm = make_llm(cfg)
    embedder = make_embedder(cfg)
    ex = Extractor(llm, cfg)
    skills = ex.extract(episode)

    added = 0
    for skill in skills:
        embedding = embedder.embed_one(skill.activation)
        try:
            store.add_skill(skill, embedding=embedding)
            added += 1
        except Exception:
            pass
    console.print(f"[green]Extracted {added} new skills from episode {episode_id}[/green]")


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
        console.print(
            f"[red]Ambiguous: {len(matches)} skills start with {id_or_prefix!r}[/red]"
        )
        raise typer.Exit(1)
    return matches[0]


def _skill_to_dict(s: Skill) -> dict:
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


if __name__ == "__main__":
    app()
