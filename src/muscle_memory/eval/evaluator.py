"""Evaluation logic: outcome accuracy, retrieval quality, skill impact."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from muscle_memory.db import Store
from muscle_memory.models import Outcome
from muscle_memory.outcomes import infer_outcome

console = Console()


# ------------------------------------------------------------------
# Outcome evaluation
# ------------------------------------------------------------------


@dataclass
class OutcomeEvalResult:
    total: int = 0
    agreement: int = 0
    matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    disagreements: list[dict] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        return self.agreement / self.total if self.total else 0.0

    def precision(self, cls: str) -> float:
        predicted = sum(self.matrix.get(cls, {}).values())
        if not predicted:
            return 0.0
        return self.matrix.get(cls, {}).get(cls, 0) / predicted

    def recall(self, cls: str) -> float:
        actual = sum(row.get(cls, 0) for row in self.matrix.values())
        if not actual:
            return 0.0
        return self.matrix.get(cls, {}).get(cls, 0) / actual


def evaluate_outcomes(store: Store) -> OutcomeEvalResult:
    """Compare heuristic outcomes against human labels."""
    labels = store.get_eval_labels("outcome")
    if not labels:
        return OutcomeEvalResult()

    classes = ["success", "failure", "unknown"]
    matrix: dict[str, dict[str, int]] = {
        pred: {actual: 0 for actual in classes} for pred in classes
    }
    disagreements: list[dict] = []
    agreement = 0
    evaluated = 0

    for label in labels:
        ep = store.get_episode(label.episode_id)
        if ep is None:
            continue
        evaluated += 1

        # Re-run heuristic with current code
        signal = infer_outcome(
            ep.trajectory,
            user_followup=ep.trajectory.user_followup,
            any_skills_activated=bool(ep.activated_skills),
        )
        predicted = signal.outcome.value
        actual = label.human_outcome

        if predicted in matrix and actual in matrix[predicted]:
            matrix[predicted][actual] += 1

        if predicted == actual:
            agreement += 1
        else:
            disagreements.append({
                "episode_id": label.episode_id,
                "prompt": ep.user_prompt[:60],
                "predicted": predicted,
                "actual": actual,
                "reasons": signal.reasons,
            })

    return OutcomeEvalResult(
        total=evaluated,
        agreement=agreement,
        matrix=matrix,
        disagreements=disagreements,
    )


def render_outcome_eval(result: OutcomeEvalResult) -> None:
    """Print outcome eval report."""
    if result.total == 0:
        console.print(
            "[dim]No outcome labels yet. Run [bold]mm eval label --outcome[/bold] first.[/dim]"
        )
        return

    console.print(f"[bold]Outcome Evaluation[/bold] ({result.total} labeled episodes)\n")

    # Confusion matrix
    classes = ["success", "failure", "unknown"]
    table = Table(title="Confusion Matrix (rows=predicted, cols=actual)")
    table.add_column("", width=12)
    for cls in classes:
        table.add_column(cls, justify="right", width=10)

    for pred in classes:
        row_values = []
        for actual in classes:
            count = result.matrix.get(pred, {}).get(actual, 0)
            if pred == actual and count > 0:
                row_values.append(f"[green]{count}[/green]")
            elif pred != actual and count > 0:
                row_values.append(f"[red]{count}[/red]")
            else:
                row_values.append(f"[dim]{count}[/dim]")
        table.add_row(pred, *row_values)
    console.print(table)

    # Metrics
    console.print(
        f"\n  [bold]Agreement:[/bold]          {result.agreement_rate:.1%}"
        f"  ({result.agreement}/{result.total})"
    )
    for cls in ["success", "failure"]:
        p = result.precision(cls)
        r = result.recall(cls)
        console.print(
            f"  [bold]Precision({cls}):[/bold]  {p:.3f}"
            f"    [bold]Recall({cls}):[/bold]  {r:.3f}"
        )

    # False positive rate (predicted success, actually failure)
    fp = result.matrix.get("success", {}).get("failure", 0)
    total_pred_success = sum(result.matrix.get("success", {}).values())
    fp_rate = fp / total_pred_success if total_pred_success else 0.0
    console.print(f"  [bold]False positive rate:[/bold] {fp_rate:.3f}  ({fp} episodes)")

    # Disagreements
    if result.disagreements:
        console.print(f"\n[bold]Disagreements ({len(result.disagreements)}):[/bold]")
        for d in result.disagreements[:10]:
            console.print(
                f"  [dim]{d['episode_id'][:8]}[/dim]  "
                f"heuristic=[yellow]{d['predicted']}[/yellow]  "
                f"human=[cyan]{d['actual']}[/cyan]  "
                f"\"{d['prompt']}\""
            )
        if len(result.disagreements) > 10:
            console.print(f"  [dim]... and {len(result.disagreements) - 10} more[/dim]")


# ------------------------------------------------------------------
# Impact evaluation
# ------------------------------------------------------------------


@dataclass
class GroupMetrics:
    count: int = 0
    successes: int = 0
    failures: int = 0
    unknowns: int = 0
    total_reward: float = 0.0
    total_tool_calls: int = 0
    total_errors: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.count if self.count else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.count if self.count else 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.count if self.count else 0.0

    @property
    def avg_tool_calls(self) -> float:
        return self.total_tool_calls / self.count if self.count else 0.0

    @property
    def avg_error_rate(self) -> float:
        if self.total_tool_calls == 0:
            return 0.0
        return self.total_errors / self.total_tool_calls


@dataclass
class SkillImpact:
    skill_id: str
    activation: str
    episodes: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


@dataclass
class ImpactEvalResult:
    with_skills: GroupMetrics = field(default_factory=GroupMetrics)
    without_skills: GroupMetrics = field(default_factory=GroupMetrics)
    per_skill: list[SkillImpact] = field(default_factory=list)


def evaluate_impact(store: Store) -> ImpactEvalResult:
    """Compare episodes with skill activation vs. without."""
    episodes = store.list_episodes(limit=10_000)

    with_skills = GroupMetrics()
    without_skills = GroupMetrics()
    skill_episodes: dict[str, SkillImpact] = {}

    for ep in episodes:
        tc_count = ep.trajectory.num_tool_calls()
        err_count = len(ep.trajectory.errored_tool_calls())

        if ep.activated_skills:
            grp = with_skills
            # Track per-skill
            for sid in ep.activated_skills:
                if sid not in skill_episodes:
                    skill = store.get_skill(sid)
                    activation = skill.activation[:50] if skill else "(deleted)"
                    skill_episodes[sid] = SkillImpact(skill_id=sid, activation=activation)
                skill_episodes[sid].episodes += 1
                if ep.outcome == Outcome.SUCCESS:
                    skill_episodes[sid].successes += 1
        else:
            grp = without_skills

        grp.count += 1
        grp.total_reward += ep.reward
        grp.total_tool_calls += tc_count
        grp.total_errors += err_count
        if ep.outcome == Outcome.SUCCESS:
            grp.successes += 1
        elif ep.outcome == Outcome.FAILURE:
            grp.failures += 1
        else:
            grp.unknowns += 1

    per_skill = sorted(
        [s for s in skill_episodes.values() if s.episodes >= 3],
        key=lambda s: s.success_rate,
        reverse=True,
    )

    return ImpactEvalResult(
        with_skills=with_skills,
        without_skills=without_skills,
        per_skill=per_skill,
    )


def render_impact_eval(result: ImpactEvalResult) -> None:
    """Print impact eval report."""
    ws = result.with_skills
    wo = result.without_skills
    total = ws.count + wo.count

    if total == 0:
        console.print("[dim]No episodes recorded yet.[/dim]")
        return

    console.print(f"[bold]Impact Evaluation[/bold] ({total} episodes)\n")

    table = Table(title="With Skills vs. Without")
    table.add_column("Metric", width=18)
    table.add_column(f"With ({ws.count})", justify="right", width=14)
    table.add_column(f"Without ({wo.count})", justify="right", width=14)
    table.add_column("Delta", justify="right", width=10)

    def _row(name: str, w: float, o: float, fmt: str = ".1%", delta_fmt: str | None = None) -> None:
        delta = w - o
        color = "green" if delta > 0 else "red" if delta < 0 else "dim"
        dfmt = delta_fmt or f"+{fmt}"
        table.add_row(
            name,
            f"{w:{fmt}}",
            f"{o:{fmt}}",
            f"[{color}]{delta:{dfmt}}[/{color}]",
        )

    _row("Success rate", ws.success_rate, wo.success_rate)
    _row("Failure rate", ws.failure_rate, wo.failure_rate)
    _row("Avg reward", ws.avg_reward, wo.avg_reward, fmt=".2f", delta_fmt="+.2f")
    _row("Avg tool calls", ws.avg_tool_calls, wo.avg_tool_calls, fmt=".1f")
    _row("Avg error rate", ws.avg_error_rate, wo.avg_error_rate)

    console.print(table)

    if ws.count < 10 or wo.count < 10:
        console.print(
            "\n[yellow]Warning: small sample size — interpret with caution.[/yellow]"
        )

    # Per-skill breakdown
    if result.per_skill:
        baseline = wo.success_rate
        console.print(f"\n[bold]Per-Skill Impact[/bold] (≥3 activations, baseline={baseline:.1%})")

        sk_table = Table(box=None, show_header=True, padding=(0, 1))
        sk_table.add_column("id", style="dim", width=10)
        sk_table.add_column("activation", width=40)
        sk_table.add_column("rate", justify="right")
        sk_table.add_column("vs baseline", justify="right")

        for si in result.per_skill:
            delta = si.success_rate - baseline
            color = "green" if delta > 0 else "red" if delta < 0 else "dim"
            tag = "  [red]HARMFUL[/red]" if delta < -0.1 else ""
            sk_table.add_row(
                si.skill_id[:8],
                si.activation,
                f"{si.successes}/{si.episodes} ({si.success_rate:.0%})",
                f"[{color}]{delta:+.1%}[/{color}]{tag}",
            )
        console.print(sk_table)
