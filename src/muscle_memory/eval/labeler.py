"""Interactive labeling workflows for the eval system."""

from __future__ import annotations

import re
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from muscle_memory.db import Store
from muscle_memory.eval import EvalLabel
from muscle_memory.models import Episode, Outcome
from muscle_memory.outcomes import infer_outcome

console = Console()

_XML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_xml_tags(text: str) -> str:
    """Remove XML/HTML-style tags from text for cleaner display."""
    return _XML_TAG_RE.sub("", text).strip()


def label_outcomes_interactive(
    store: Store,
    *,
    limit: int = 10,
) -> int:
    """Run an interactive outcome-labeling session. Returns count of labels added."""
    episodes = store.list_unlabeled_episodes(limit=limit)

    if not episodes:
        console.print("[dim]No unlabeled episodes remaining.[/dim]")
        return 0

    total = store.count_episodes()
    labeled = store.count_eval_labels("outcome")
    console.print(
        f"[bold]Outcome labeling[/bold]  "
        f"({labeled}/{total} labeled, {len(episodes)} to review)\n"
    )
    console.print(
        "[dim]Keys: [bold]s[/bold]=success  [bold]f[/bold]=failure  "
        "[bold]u[/bold]=unknown  [bold]?[/bold]=skip  [bold]q[/bold]=quit[/dim]\n"
    )

    added = 0
    for i, ep in enumerate(episodes, 1):
        signal = infer_outcome(
            ep.trajectory,
            user_followup=ep.trajectory.user_followup,
            any_skills_activated=bool(ep.activated_skills),
        )

        _render_episode(ep, signal, i, len(episodes))

        while True:
            try:
                key = console.input("[bold]Label:[/bold] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Quit.[/dim]")
                return added

            if key == "q":
                console.print(f"[dim]Labeled {added} episodes this session.[/dim]")
                return added
            if key == "?":
                console.print("[dim]Skipped.[/dim]\n")
                break
            if key in ("s", "f", "u"):
                human = {"s": "success", "f": "failure", "u": "unknown"}[key]
                label = EvalLabel(
                    id=uuid.uuid4().hex[:16],
                    label_type="outcome",
                    episode_id=ep.id,
                    heuristic_outcome=signal.outcome.value,
                    human_outcome=human,
                )
                store.add_eval_label(label)
                added += 1

                match = "agree" if human == signal.outcome.value else "DISAGREE"
                color = "green" if match == "agree" else "red"
                console.print(
                    f"  [{color}]{match}[/{color}]: "
                    f"human={human}, heuristic={signal.outcome.value}\n"
                )
                break
            console.print("[dim]  Press s/f/u/?/q[/dim]")

    console.print(f"[green]Done.[/green] Labeled {added} episodes this session.")
    return added


def _render_episode(ep: Episode, signal, idx: int, total: int) -> None:
    """Display an episode for labeling."""
    from muscle_memory.cli import _format_outcome, _relative_time, _short_id

    # Header
    console.print(
        f"[bold]Episode {_short_id(ep.id)}[/bold]  "
        f"({_relative_time(ep.started_at)})  "
        f"[dim]{idx}/{total}[/dim]"
    )

    # Prompt — strip Claude Code internal XML tags for readability
    prompt_clean = _strip_xml_tags(ep.user_prompt)
    prompt_display = prompt_clean[:200]
    if len(prompt_clean) > 200:
        prompt_display += "…"
    console.print(f"[bold]Prompt:[/bold] {prompt_display}")

    # Heuristic result
    console.print(
        f"[bold]Heuristic:[/bold] {_format_outcome(signal.outcome)} "
        f"(reward {signal.reward:+.2f})"
    )
    if signal.reasons:
        for r in signal.reasons:
            console.print(f"  [dim]• {r}[/dim]")

    # Trajectory summary — last 8 tool calls
    calls = ep.trajectory.tool_calls
    display_calls = calls[-8:] if len(calls) > 8 else calls
    if len(calls) > 8:
        console.print(f"\n[dim]… {len(calls) - 8} earlier tool calls …[/dim]")

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("idx", style="dim", width=5)
    table.add_column("tool", width=12)
    table.add_column("detail")

    offset = len(calls) - len(display_calls)
    for j, tc in enumerate(display_calls):
        idx_str = f"[{offset + j}]"
        cmd = str(tc.arguments.get("command", ""))[:60] if tc.arguments else ""
        detail = cmd or tc.name

        if tc.is_error():
            result_preview = (tc.error or "")[:80]
            result_str = f"[red]ERR: {result_preview}[/red]"
        else:
            result_preview = (tc.result or "")[:80]
            result_str = f"[dim]{result_preview}[/dim]"

        table.add_row(idx_str, tc.name, f"{detail}\n  {result_str}")

    console.print(table)

    # User followup
    if ep.trajectory.user_followup:
        followup = ep.trajectory.user_followup[:200]
        console.print(f"\n[bold]User followup:[/bold] {followup}")

    # Skills activated
    if ep.activated_skills:
        console.print(
            f"[dim]Skills activated: {len(ep.activated_skills)}[/dim]"
        )

    console.print()
