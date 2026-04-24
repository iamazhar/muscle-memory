"""Personal compounding loop helpers."""

from __future__ import annotations

from muscle_memory.db import Store
from muscle_memory.models import ActivationRecord, DeliveryMode, TaskRecord
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


def format_context(hits: list[RetrievedSkill]) -> str:
    """Format playbooks for manual/Codex use."""
    if not hits:
        return ""
    lines = [
        "<muscle_memory>",
        "Use the relevant playbook below only if its activation matches the task.",
        "If it matches, execute the steps directly: run commands, edit files, and verify.",
        "If it does not match, ignore it and proceed normally.",
        "",
    ]
    for index, hit in enumerate(hits, start=1):
        skill = hit.skill
        lines.extend(
            [
                f"## Playbook {index}",
                f"Activate when: {skill.activation}",
                "Steps:",
                skill.execution,
                f"Done when: {skill.termination}",
            ]
        )
        if skill.tool_hints:
            lines.append(f"Preferred tools: {', '.join(skill.tool_hints)}")
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
    per_hit_tokens = max(1, context_token_count // len(hits))
    records: list[ActivationRecord] = []
    for hit in hits:
        record = ActivationRecord(
            task_id=task.id,
            skill_id=hit.skill.id,
            distance=hit.distance,
            final_rank=hit.final_rank,
            delivery_mode=delivery_mode,
            injected_token_count=per_hit_tokens,
        )
        store.add_activation(record)
        records.append(record)
    return records
