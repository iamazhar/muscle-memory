"""`mm` — command-line interface for muscle-memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
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

eval_app = typer.Typer(help="Evaluate outcome detection, retrieval, and skill impact.")
app.add_typer(eval_app, name="eval")


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
    console.print(
        "[yellow]muscle-memory paused.[/yellow] No retrieval or extraction until `mm resume`."
    )


@app.command()
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


@app.command()
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
            data.append({
                "id": _short_id(ep.id),
                "session_id": ep.session_id or "",
                "prompt": ep.user_prompt[:80],
                "outcome": ep.outcome.value,
                "reward": ep.reward,
                "tool_calls": ep.trajectory.num_tool_calls(),
                "skills_activated": len(ep.activated_skills),
                "started_at": ep.started_at.isoformat() if ep.started_at else None,
            })
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


@app.command()
def stats(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Summarize the skill store with actionable metrics."""
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
    reuse_rate = (total_successes / total_invocations) if total_invocations else 0.0

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
    from muscle_memory.refine import should_auto_refine

    need_refine = [s for s in skills if should_auto_refine(s)]
    at_risk = [s for s in skills if s.invocations >= 5 and s.score <= 0.2]
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

    # Top and struggling skills
    top_skills = [s for s in skills if s.invocations >= 2][:3]
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
                "stale": len(stale),
                "unknown_rate": unknown_rate,
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
        f" · {by_maturity[Maturity.ESTABLISHED]} established"
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
    if attention_items == 0:
        console.print("  [green]No issues detected.[/green]")

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
            user_followup=ep.trajectory.user_followup,
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


@eval_app.command("credits")
def eval_credits(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Evaluate whether skill credits are deserved based on labels."""
    from muscle_memory.eval.evaluator import evaluate_credits, render_credit_eval

    cfg = _load_config()
    store = _open_store(cfg)
    result = evaluate_credits(store)

    if as_json:
        import json as _json

        data = {
            "total": result.total,
            "precision": result.precision,
            "deserved": result.deserved,
            "undeserved": result.undeserved,
            "per_skill": [
                {
                    "skill_id": s.skill_id,
                    "activation": s.activation,
                    "precision": s.precision,
                    "deserved": s.deserved,
                    "total": s.total,
                }
                for s in result.per_skill
            ],
        }
        typer.echo(_json.dumps(data, indent=2))
        return

    render_credit_eval(result)


@eval_app.command("impact")
def eval_impact(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Compare skill-activated vs. non-activated episode outcomes."""
    from muscle_memory.eval.evaluator import evaluate_impact, render_impact_eval

    cfg = _load_config()
    store = _open_store(cfg)
    result = evaluate_impact(store)

    if as_json:
        import json as _json

        data = {
            "with_skills": {
                "count": result.with_skills.count,
                "success_rate": result.with_skills.success_rate,
                "failure_rate": result.with_skills.failure_rate,
                "avg_reward": result.with_skills.avg_reward,
                "avg_tool_calls": result.with_skills.avg_tool_calls,
                "avg_error_rate": result.with_skills.avg_error_rate,
            },
            "without_skills": {
                "count": result.without_skills.count,
                "success_rate": result.without_skills.success_rate,
                "failure_rate": result.without_skills.failure_rate,
                "avg_reward": result.without_skills.avg_reward,
                "avg_tool_calls": result.without_skills.avg_tool_calls,
                "avg_error_rate": result.without_skills.avg_error_rate,
            },
            "per_skill": [
                {
                    "skill_id": s.skill_id,
                    "activation": s.activation,
                    "success_rate": s.success_rate,
                    "successes": s.successes,
                    "episodes": s.episodes,
                }
                for s in result.per_skill
            ],
        }
        typer.echo(_json.dumps(data, indent=2))
        return

    render_impact_eval(result)


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
    entries, path = build_benchmark(store, output_path=out)
    console.print(
        f"[green]Built benchmark:[/green] {len(entries)} entries -> {path}"
    )


@eval_app.command("run")
def eval_run(
    benchmark: str = typer.Option(None, "--benchmark", "-b", help="Path to benchmark JSON."),
) -> None:
    """Re-score against a frozen benchmark and show diffs."""
    from pathlib import Path as _Path

    from rich.panel import Panel as _Panel

    from muscle_memory.eval.benchmark import run_benchmark

    cfg = _load_config()
    store = _open_store(cfg)

    if benchmark:
        path = _Path(benchmark)
    else:
        path = cfg.db_path.parent / "benchmark.json"

    if not path.exists():
        console.print(
            f"[red]No benchmark at {path}.[/red] Run [bold]mm eval build[/bold] first."
        )
        return

    result = run_benchmark(store, path)

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

    # Health
    health = evaluate_health(store)
    render_health_report(health)

    console.print()

    # Impact
    impact = evaluate_impact(store)
    render_impact_eval(impact)


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
