"""Evaluation logic: credit accuracy and skill impact."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from muscle_memory.db import Store
from muscle_memory.models import Episode, Outcome

console = Console()


# ------------------------------------------------------------------
# Credit evaluation — are skill credits deserved?
# ------------------------------------------------------------------


@dataclass
class SkillCreditStats:
    skill_id: str
    activation: str
    total: int = 0
    deserved: int = 0
    undeserved: int = 0

    @property
    def precision(self) -> float:
        return self.deserved / self.total if self.total else 0.0


@dataclass
class CreditEvalResult:
    total: int = 0
    deserved: int = 0
    undeserved: int = 0
    per_skill: list[SkillCreditStats] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.deserved / self.total if self.total else 0.0


def evaluate_credits(store: Store) -> CreditEvalResult:
    """Evaluate whether skill credits were deserved based on human labels."""
    labels = store.get_eval_labels("credit")
    if not labels:
        return CreditEvalResult()

    total = 0
    deserved = 0
    undeserved = 0
    by_skill: dict[str, SkillCreditStats] = {}

    for label in labels:
        if label.human_outcome not in ("deserved", "undeserved"):
            continue
        total += 1

        is_deserved = label.human_outcome == "deserved"
        if is_deserved:
            deserved += 1
        else:
            undeserved += 1

        # Per-skill tracking
        sid = label.skill_id
        if sid and sid not in by_skill:
            skill = store.get_skill(sid)
            activation = skill.activation[:60] if skill else "(deleted)"
            by_skill[sid] = SkillCreditStats(skill_id=sid, activation=activation)
        if sid:
            by_skill[sid].total += 1
            if is_deserved:
                by_skill[sid].deserved += 1
            else:
                by_skill[sid].undeserved += 1

    per_skill = sorted(by_skill.values(), key=lambda s: s.precision)
    return CreditEvalResult(
        total=total,
        deserved=deserved,
        undeserved=undeserved,
        per_skill=per_skill,
    )


def render_credit_eval(result: CreditEvalResult) -> None:
    """Print credit eval report."""
    if result.total == 0:
        console.print(
            "[dim]No credit labels yet. Run [bold]mm eval label[/bold] first.[/dim]"
        )
        return

    console.print(f"[bold]Credit Evaluation[/bold] ({result.total} labeled)\n")

    # Overall precision
    color = "green" if result.precision >= 0.8 else "yellow" if result.precision >= 0.6 else "red"
    console.print(
        f"  [bold]Credit precision:[/bold] [{color}]{result.precision:.1%}[/{color}]"
        f"  ({result.deserved} deserved, {result.undeserved} undeserved)"
    )

    if result.precision < 0.8:
        console.print(
            f"\n  [yellow]The heuristic is giving undeserved credit "
            f"{result.undeserved}/{result.total} times.[/yellow]\n"
            f"  [dim]This inflates scores for skills that didn't actually help.[/dim]"
        )

    # Per-skill breakdown
    if result.per_skill:
        console.print(f"\n[bold]Per-Skill Credit Precision[/bold]")

        table = Table(box=None, show_header=True, padding=(0, 1))
        table.add_column("id", style="dim", width=10)
        table.add_column("precision", justify="right", width=10)
        table.add_column("labels", justify="right", width=8)
        table.add_column("activation", width=50)

        for s in result.per_skill:
            p = s.precision
            color = "green" if p >= 0.8 else "yellow" if p >= 0.5 else "red"
            tag = ""
            if p < 0.5 and s.total >= 3:
                tag = " [red]INFLATED[/red]"
            table.add_row(
                s.skill_id[:8],
                f"[{color}]{p:.0%}[/{color}]{tag}",
                f"{s.deserved}/{s.total}",
                s.activation,
            )
        console.print(table)


# ------------------------------------------------------------------
# Health report — automated playbook scoring
# ------------------------------------------------------------------


@dataclass
class PlaybookHealth:
    skill_id: str
    activation: str
    activations: int = 0
    avg_relevance: float = 0.0
    avg_adherence: float = 0.0
    correct: int = 0
    incorrect: int = 0
    needs_review: int = 0
    step_count: int = 0  # how many steps the playbook prescribes
    matched_step_total: int = 0  # total matched steps across all activations

    @property
    def health_pct(self) -> float:
        if self.activations == 0:
            return 0.0
        return self.correct / self.activations


@dataclass
class HealthReport:
    total_activations: int = 0
    healthy_pct: float = 0.0
    avg_relevance: float = 0.0
    avg_adherence: float = 0.0
    per_skill: list[PlaybookHealth] = field(default_factory=list)
    needs_review_count: int = 0


def evaluate_health(store: Store) -> HealthReport:
    """Score all skill activations and produce a health report.

    Runs the three automated scorers (relevance, adherence, correctness)
    on every (episode, skill) activation pair.
    """
    from muscle_memory.eval.scorers import (
        load_activation_distances,
        score_adherence,
        score_correctness,
        score_relevance,
    )

    episodes = store.list_episodes(limit=10_000)

    # Deduplicate by session
    by_session: dict[str, Episode] = {}
    for ep in episodes:
        sid = ep.session_id or ep.id
        existing = by_session.get(sid)
        if existing is None or ep.trajectory.num_tool_calls() > existing.trajectory.num_tool_calls():
            by_session[sid] = ep

    by_skill: dict[str, PlaybookHealth] = {}
    all_relevances: list[float] = []
    all_adherences: list[float] = []
    total = 0
    healthy = 0
    needs_review = 0

    for ep in by_session.values():
        if not ep.activated_skills:
            continue

        # Load stored distances from sidecar
        distances = load_activation_distances(store.db_path, ep.session_id or "")

        for skill_id in set(ep.activated_skills):
            skill = store.get_skill(skill_id)
            if skill is None:
                continue

            rel = score_relevance(
                store, ep, skill_id,
                stored_distance=distances.get(skill_id),
            )
            adh = score_adherence(skill, ep.trajectory)
            cor = score_correctness(adh, ep.outcome)

            total += 1
            all_relevances.append(rel.score)
            all_adherences.append(adh.score)

            if rel.score >= 0.5 and adh.score >= 0.5 and cor.verdict == "correct":
                healthy += 1
            if cor.verdict == "needs_review":
                needs_review += 1

            if skill_id not in by_skill:
                by_skill[skill_id] = PlaybookHealth(
                    skill_id=skill_id,
                    activation=skill.activation[:60],
                    step_count=adh.total_steps,
                )
            ph = by_skill[skill_id]
            ph.activations += 1
            ph.matched_step_total += len(adh.matched_steps)
            # Running average
            ph.avg_relevance += (rel.score - ph.avg_relevance) / ph.activations
            ph.avg_adherence += (adh.score - ph.avg_adherence) / ph.activations
            if cor.verdict == "correct":
                ph.correct += 1
            elif cor.verdict == "incorrect":
                ph.incorrect += 1
            else:
                ph.needs_review += 1

    per_skill = sorted(by_skill.values(), key=lambda p: p.health_pct, reverse=True)

    return HealthReport(
        total_activations=total,
        healthy_pct=healthy / total if total else 0.0,
        avg_relevance=sum(all_relevances) / len(all_relevances) if all_relevances else 0.0,
        avg_adherence=sum(all_adherences) / len(all_adherences) if all_adherences else 0.0,
        per_skill=per_skill,
        needs_review_count=needs_review,
    )


def render_health_report(report: HealthReport) -> None:
    """Print the health dashboard."""
    if report.total_activations == 0:
        console.print("[dim]No skill activations found. Run some sessions first.[/dim]")
        return

    console.print(f"[bold]Playbook Health[/bold] ({report.total_activations} activations)\n")

    # Overall metrics
    total_steps = sum(ph.step_count for ph in report.per_skill)
    total_matched = sum(ph.matched_step_total for ph in report.per_skill)

    h_color = "green" if report.healthy_pct >= 0.7 else "yellow" if report.healthy_pct >= 0.5 else "red"
    console.print(
        f"  [bold]Healthy:[/bold] [{h_color}]{report.healthy_pct:.0%}[/{h_color}]"
        f"  [bold]Avg relevance:[/bold] {report.avg_relevance:.2f}"
        f"  [bold]Avg adherence:[/bold] {report.avg_adherence:.2f}"
    )
    if total_steps:
        console.print(
            f"  [bold]Steps executed:[/bold] {total_matched} / "
            f"{total_steps * report.total_activations // len(report.per_skill)} possible"
            f"  [dim](each followed step = a problem solved without trial-and-error)[/dim]"
        )
    if report.needs_review_count:
        console.print(
            f"  [yellow]{report.needs_review_count} activations need human review[/yellow]"
            f" (run [bold]mm eval label[/bold])"
        )

    # Per-skill table
    if report.per_skill:
        console.print()
        table = Table(title="Per-Skill Health")
        table.add_column("id", style="dim", width=10)
        table.add_column("activation", width=35)
        table.add_column("acts", justify="right", width=5)
        table.add_column("steps", justify="right", width=6)
        table.add_column("adherence", justify="right", width=10)
        table.add_column("correct", justify="right", width=10)
        table.add_column("status", width=12)

        for ph in report.per_skill:
            a_color = "green" if ph.avg_adherence >= 0.5 else "red"
            h_pct = ph.health_pct

            if h_pct >= 0.7:
                status = "[green]healthy[/green]"
            elif ph.incorrect > 0:
                status = "[red]bad steps[/red]"
            elif ph.avg_adherence < 0.3:
                status = "[yellow]ignored[/yellow]"
            elif ph.needs_review > 0:
                status = "[yellow]review[/yellow]"
            else:
                status = "[dim]ok[/dim]"

            table.add_row(
                ph.skill_id[:8],
                ph.activation,
                str(ph.activations),
                str(ph.step_count),
                f"[{a_color}]{ph.avg_adherence:.2f}[/{a_color}]",
                f"{ph.correct}/{ph.activations}",
                status,
            )
        console.print(table)


# ------------------------------------------------------------------
# Impact evaluation — do skills help overall?
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
            "\n[yellow]Warning: small sample size.[/yellow]"
        )

    if result.per_skill:
        baseline = wo.success_rate
        console.print(f"\n[bold]Per-Skill Impact[/bold] (>=3 activations, baseline={baseline:.1%})")

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
