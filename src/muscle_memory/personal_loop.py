"""Personal compounding loop helpers."""

from __future__ import annotations

from dataclasses import dataclass

from muscle_memory.db import Store
from muscle_memory.models import (
    ActivationRecord,
    DeliveryMode,
    EvidenceConfidence,
    MeasurementRecord,
    Outcome,
    TaskRecord,
)
from muscle_memory.prompt_cleaning import clean_prompt_text
from muscle_memory.retriever import RetrievedSkill


def count_text_tokens(text: str) -> int:
    """Cheap deterministic token estimate for locally formatted context."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def capture_task(
    store: Store,
    *,
    raw_prompt: str,
    harness: str,
    project_path: str | None,
    session_id: str | None = None,
    cleaned_prompt: str | None = None,
) -> TaskRecord:
    cleaned = cleaned_prompt if cleaned_prompt is not None else clean_prompt_text(raw_prompt)
    task = TaskRecord(
        raw_prompt=raw_prompt,
        cleaned_prompt=cleaned or raw_prompt.strip() or "(unknown)",
        harness=harness,
        project_path=project_path,
        session_id=session_id,
    )
    store.add_task(task)
    return task


def _skill_title(activation: str, max_len: int = 60) -> str:
    s = activation.strip()
    if s.lower().startswith("when "):
        s = s[5:]
    for stop in (". ", ", "):
        i = s.find(stop)
        if 10 < i < max_len:
            s = s[:i]
            break
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def format_context(hits: list[RetrievedSkill]) -> str:
    """Format retrieved skills as imperative playbooks to execute."""
    if not hits:
        return ""
    titles = [_skill_title(hit.skill.activation) for hit in hits]
    titles_list = " | ".join(f'"{title}"' for title in titles)

    lines = [
        "<muscle_memory>",
        "These are verified playbooks extracted from past successful sessions",
        "in this project. For each playbook below, if the `Activate when`",
        "condition clearly matches the user's current situation, **EXECUTE the",
        "Steps directly**: run the commands, make the edits, verify the result.",
        "Do not just describe the steps to the user — actually perform them.",
        "The user wants the problem fixed, not a list of instructions.",
        "",
        "If a playbook's `Activate when` clearly does NOT fit the current task,",
        "ignore it and proceed normally.",
        "",
        "### Visibility protocol (required)",
        "",
        "Begin your response with ONE line in exactly this format so the user",
        "can see which playbook fired:",
        "",
        "> 🧠 **muscle-memory**: executing playbook — <title>",
        "",
        f"Where `<title>` is one of: {titles_list}",
        "",
        "If NONE of the playbooks apply to the current task, do NOT emit any",
        "muscle-memory marker. Just proceed normally with the user's request.",
        "Do not explain muscle-memory or discuss the playbook metadata.",
        "",
    ]
    for index, (hit, title) in enumerate(zip(hits, titles), start=1):
        skill = hit.skill
        lines.append(
            f'## Playbook {index} — "{title}"'
            f" · {skill.maturity.value}"
            f" · {skill.successes}/{skill.invocations} successes"
        )
        lines.append(f"**Activate when:** {skill.activation}")
        lines.append("**Steps (execute in order):**")
        lines.append(skill.execution)
        lines.append(f"**Done when:** {skill.termination}")
        if skill.tool_hints:
            lines.append(f"**Preferred tools:** {', '.join(skill.tool_hints)}")
        lines.append("")
    lines.append("</muscle_memory>")
    return "\n".join(lines)


def record_activations(
    store: Store,
    *,
    task: TaskRecord,
    hits: list[RetrievedSkill],
    delivery_mode: DeliveryMode,
    context_token_count: int,
) -> list[ActivationRecord]:
    if not hits:
        return []
    base_tokens = context_token_count // len(hits)
    remainder = context_token_count % len(hits)
    records: list[ActivationRecord] = []
    for index, hit in enumerate(hits):
        record = ActivationRecord(
            task_id=task.id,
            skill_id=hit.skill.id,
            distance=hit.distance,
            final_rank=hit.final_rank,
            delivery_mode=delivery_mode,
            injected_token_count=base_tokens + (1 if index < remainder else 0),
        )
        store.add_activation(record)
        records.append(record)
    return records


@dataclass(frozen=True)
class ProofMetrics:
    confidence: EvidenceConfidence
    comparable_tasks: int
    assisted_tasks: int
    unassisted_tasks: int
    assisted_success_rate: float | None
    unassisted_success_rate: float | None
    outcome_lift: float | None
    token_reduction: float | None
    token_samples: int
    unknown_outcomes: int


def compute_proof_metrics(store: Store) -> ProofMetrics:
    measurements = [m for m in store.list_measurements(limit=None) if m.comparable]
    task_ids_with_activation = store.list_task_ids_with_activations()
    assisted = [m for m in measurements if m.task_id in task_ids_with_activation]
    unassisted = [m for m in measurements if m.task_id not in task_ids_with_activation]
    assisted_known = [m for m in assisted if m.outcome is not Outcome.UNKNOWN]
    unassisted_known = [m for m in unassisted if m.outcome is not Outcome.UNKNOWN]

    assisted_success_rate = _success_rate(assisted_known)
    unassisted_success_rate = _success_rate(unassisted_known)
    outcome_lift = (
        assisted_success_rate - unassisted_success_rate
        if assisted_success_rate is not None and unassisted_success_rate is not None
        else None
    )

    assisted_tokens = [
        token_total for m in assisted if (token_total := _measured_token_total(m)) is not None
    ]
    unassisted_tokens = [
        token_total
        for m in unassisted
        if (token_total := _measured_token_total(m)) is not None
    ]
    token_reduction = None
    if assisted_tokens and unassisted_tokens:
        assisted_avg = sum(assisted_tokens) / len(assisted_tokens)
        unassisted_avg = sum(unassisted_tokens) / len(unassisted_tokens)
        if unassisted_avg > 0:
            token_reduction = (unassisted_avg - assisted_avg) / unassisted_avg

    token_samples = len(assisted_tokens) + len(unassisted_tokens)
    unknowns = sum(1 for m in measurements if m.outcome is Outcome.UNKNOWN)
    confidence = _proof_confidence(
        comparable_count=len(measurements),
        assisted_known_count=len(assisted_known),
        unassisted_known_count=len(unassisted_known),
        assisted_token_samples=len(assisted_tokens),
        unassisted_token_samples=len(unassisted_tokens),
        outcome_lift=outcome_lift,
        token_reduction=token_reduction,
        unknowns=unknowns,
    )

    return ProofMetrics(
        confidence=confidence,
        comparable_tasks=len(measurements),
        assisted_tasks=len(assisted),
        unassisted_tasks=len(unassisted),
        assisted_success_rate=assisted_success_rate,
        unassisted_success_rate=unassisted_success_rate,
        outcome_lift=outcome_lift,
        token_reduction=token_reduction,
        token_samples=token_samples,
        unknown_outcomes=unknowns,
    )


def _success_rate(measurements: list[MeasurementRecord]) -> float | None:
    if not measurements:
        return None
    successes = sum(1 for m in measurements if m.outcome is Outcome.SUCCESS)
    return successes / len(measurements)


def _measured_token_total(measurement: MeasurementRecord) -> int | None:
    if measurement.input_tokens is None or measurement.output_tokens is None:
        return None
    return measurement.input_tokens + measurement.output_tokens


def _proof_confidence(
    *,
    comparable_count: int,
    assisted_known_count: int,
    unassisted_known_count: int,
    assisted_token_samples: int,
    unassisted_token_samples: int,
    outcome_lift: float | None,
    token_reduction: float | None,
    unknowns: int,
) -> EvidenceConfidence:
    if (
        comparable_count >= 50
        and assisted_known_count >= 10
        and unassisted_known_count >= 10
        and assisted_token_samples >= 10
        and unassisted_token_samples >= 10
        and outcome_lift is not None
        and token_reduction is not None
        and unknowns / comparable_count <= 0.2
    ):
        return EvidenceConfidence.HIGH
    if (
        comparable_count >= 10
        and assisted_known_count >= 3
        and unassisted_known_count >= 3
        and outcome_lift is not None
    ):
        return EvidenceConfidence.MEDIUM
    return EvidenceConfidence.LOW


def add_measurement_for_task(
    store: Store,
    *,
    task: TaskRecord,
    outcome: Outcome,
    reason: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    injected_skill_tokens: int = 0,
    tool_call_count: int = 0,
    comparable: bool = False,
) -> MeasurementRecord:
    if outcome is Outcome.UNKNOWN:
        confidence = EvidenceConfidence.LOW
    elif comparable and tool_call_count > 0:
        confidence = EvidenceConfidence.HIGH
    else:
        confidence = EvidenceConfidence.MEDIUM

    measurement = MeasurementRecord(
        task_id=task.id,
        outcome=outcome,
        confidence=confidence,
        reason=reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        injected_skill_tokens=injected_skill_tokens,
        tool_call_count=tool_call_count,
        comparable=comparable,
    )
    store.add_measurement(measurement)
    return measurement
