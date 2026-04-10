"""`mm` — command-line interface for muscle-memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
def pause() -> None:
    """Pause muscle-memory. Hooks will silently no-op until resumed."""
    cfg = _load_config()
    flag = cfg.db_path.parent / "mm.paused"
    flag.touch()
    console.print("[yellow]muscle-memory paused.[/yellow] No retrieval or extraction until `mm resume`.")


@app.command()
def resume() -> None:
    """Resume muscle-memory after a pause."""
    cfg = _load_config()
    flag = cfg.db_path.parent / "mm.paused"
    if flag.exists():
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
        "\nNext: use Claude Code as usual. Optionally seed with [bold]mm bootstrap[/bold].",
    )


@app.command("list")
def list_skills(
    maturity: Maturity | None = typer.Option(None, "--maturity", "-m"),
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

    paused = (cfg.db_path.parent / "mm.paused").exists()
    status_line = "[yellow]PAUSED[/yellow]" if paused else "[green]active[/green]"

    panel_text = (
        f"[bold]database[/bold] {cfg.db_path}\n"
        f"[bold]project root[/bold] {cfg.project_root or '(global)'}\n"
        f"[bold]status[/bold] {status_line}\n\n"
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


@app.command()
def rescore() -> None:
    """Re-run the outcome heuristic on every episode and re-credit skills.

    Use this after upgrading muscle-memory or tweaking the outcome
    heuristic — it retroactively applies the new rules to historical
    episodes so skill scores reflect current logic.
    """
    from muscle_memory.outcomes import infer_outcome
    from muscle_memory.scorer import Scorer

    cfg = _load_config()
    store = _open_store(cfg)

    # First, reset skill counters so rescoring is idempotent.
    reset_n = 0
    for skill in store.list_skills():
        skill.successes = 0
        skill.failures = 0
        # keep invocations (set by user_prompt hook at retrieval time)
        skill.recompute_score()
        skill.recompute_maturity()
        store.update_skill(skill)
        reset_n += 1

    # Iterate episodes, re-infer outcomes, re-credit skills.
    episodes = store.list_episodes(limit=10_000)
    scorer = Scorer(store, max_skills=cfg.max_skills)
    outcome_counts = {"success": 0, "failure": 0, "unknown": 0}

    for ep in episodes:
        signal = infer_outcome(
            ep.trajectory,
            any_skills_activated=bool(ep.activated_skills),
        )
        ep.outcome = signal.outcome
        ep.reward = signal.reward
        outcome_counts[signal.outcome.value] += 1
        store.update_episode_outcome(ep.id, outcome=signal.outcome, reward=signal.reward)
        scorer.credit_episode(ep)

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
    console.print(f"[green]Pruned {len(report.removed)} skills.[/green] {report.kept} remain.")


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
        True,
        "--project-only/--all-projects",
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
    if report.aborted_reason:
        console.print(
            f"\n[red]aborted:[/red] {report.aborted_reason}\n"
            "[dim]fix the underlying issue (API credits, auth, model name) "
            "and re-run `mm bootstrap`.[/dim]"
        )
        raise typer.Exit(1)
    if report.errors:
        for err in report.errors[:5]:
            console.print(f"[yellow]  {err}[/yellow]")


@app.command("extract-episode", hidden=True)
def extract_episode_cmd(episode_id: str) -> None:
    """Extract skills from a single episode (used by the async pipeline)."""
    from muscle_memory.dedup import add_skill_with_dedup
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


@app.command()
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


if __name__ == "__main__":
    app()
