"""Claude Code harness adapter."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from muscle_memory.harness.base import InstallReport
from muscle_memory.models import Scope, ToolCall, Trajectory
from muscle_memory.prompt_cleaning import clean_prompt_text

if TYPE_CHECKING:
    from muscle_memory.config import Config
    from muscle_memory.retriever import RetrievedSkill

USER_PROMPT_EVENT = "UserPromptSubmit"
STOP_EVENT = "Stop"
USER_PROMPT_CMD = "mm hook user-prompt"
STOP_CMD = "mm hook stop"

_BANG_PREFIXES = (
    "!",
    "/",
)

_SHELL_CMD_HEADS = {
    "ls",
    "cd",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "pwd",
    "git",
    "npm",
    "yarn",
    "uv",
    "pip",
    "python",
    "python3",
    "node",
    "bash",
    "sh",
    "zsh",
    "which",
    "whoami",
    "env",
    "export",
    "echo",
    "mm",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "touch",
    "chmod",
    "chflags",
    "curl",
    "wget",
    "ssh",
    "scp",
    "kubectl",
    "docker",
}


class ClaudeCodeHarness:
    name = "claude-code"

    def install(self, config: Config) -> InstallReport:
        if config.scope is Scope.PROJECT and config.project_root is None:
            raise RuntimeError(
                "Not inside a project (no .git or .claude found). Either `cd` into a project or use --scope global."
            )

        config.ensure_db_dir()
        from muscle_memory.db import Store

        Store(config.db_path, embedding_dims=config.embedding_dims)

        settings_root = config.project_root if config.scope is Scope.PROJECT else Path.home()
        assert settings_root is not None
        settings_path = settings_root / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        settings: dict[str, Any] = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Could not parse existing {settings_path}. Fix it manually or remove it before running `mm init`."
                ) from None

        original = deepcopy(settings)
        installed: list[str] = []
        already: list[str] = []

        for event, cmd in ((USER_PROMPT_EVENT, USER_PROMPT_CMD), (STOP_EVENT, STOP_CMD)):
            changed = _ensure_hook(settings, event, cmd)
            if changed:
                installed.append(event)
            else:
                already.append(event)

        if settings != original:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

        return InstallReport(
            settings_path=settings_path,
            db_path=config.db_path,
            installed_events=installed,
            already_present=already,
        )

    def uninstall(self, config: Config) -> InstallReport:
        settings_root = config.project_root if config.project_root else Path.home()
        settings_path = settings_root / ".claude" / "settings.json"
        removed: list[str] = []

        if settings_path.exists():
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            hooks = settings.get("hooks", {})
            for event in (USER_PROMPT_EVENT, STOP_EVENT):
                groups = hooks.get(event, []) or []
                filtered: list[dict[str, Any]] = []
                for group in groups:
                    if not isinstance(group, dict):
                        filtered.append(group)
                        continue
                    inner = [
                        h
                        for h in (group.get("hooks") or [])
                        if not (isinstance(h, dict) and h.get("command", "").startswith("mm hook"))
                    ]
                    if inner:
                        group["hooks"] = inner
                        filtered.append(group)
                    else:
                        removed.append(event)
                if filtered:
                    hooks[event] = filtered
                elif event in hooks:
                    del hooks[event]
            settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

        return InstallReport(
            settings_path=settings_path,
            db_path=config.db_path,
            installed_events=[],
            already_present=removed,
        )

    def extract_prompt(self, payload: dict[str, Any]) -> str:
        for key in ("prompt", "user_prompt", "message", "text"):
            val = payload.get(key)
            if isinstance(val, str):
                return val
        hook_data = payload.get("hook_event") or {}
        if isinstance(hook_data, dict):
            for key in ("prompt", "user_prompt", "message"):
                nested_val = hook_data.get(key)
                if isinstance(nested_val, str):
                    return nested_val
        return ""

    def extract_session_id(self, payload: dict[str, Any]) -> str:
        value = payload.get("session_id")
        return value if isinstance(value, str) else ""

    def extract_cwd(self, payload: dict[str, Any]) -> Path | None:
        value = payload.get("cwd")
        if isinstance(value, str) and value:
            return Path(value)
        return None

    def extract_transcript_path(self, payload: dict[str, Any]) -> Path | None:
        value = payload.get("transcript_path")
        if isinstance(value, str) and value:
            return Path(value).expanduser()
        return None

    def parse_transcript(self, path: Path) -> Trajectory:
        user_prompt = ""
        user_followups: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_turns: list[str] = []
        pending_by_id: dict[str, ToolCall] = {}

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec_type = rec.get("type")
                msg = rec.get("message") or {}

                if rec_type == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        text = clean_prompt_text(content)
                        if text:
                            if not user_prompt:
                                user_prompt = text
                            else:
                                user_followups.append(text)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id")
                                    result_content = block.get("content")
                                    is_error = block.get("is_error", False)
                                    result_text = _flatten_content(result_content)
                                    if tool_id and tool_id in pending_by_id:
                                        tc = pending_by_id[tool_id]
                                        if is_error:
                                            tc.error = result_text
                                        else:
                                            tc.result = result_text
                                elif block.get("type") == "text":
                                    text = clean_prompt_text(block.get("text", ""))
                                    if text:
                                        if not user_prompt:
                                            user_prompt = text
                                        else:
                                            user_followups.append(text)

                elif rec_type == "assistant":
                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "text":
                                assistant_turns.append(block.get("text", ""))
                            elif btype == "tool_use":
                                tc = ToolCall(
                                    name=block.get("name", "unknown"),
                                    arguments=block.get("input", {}) or {},
                                )
                                tool_calls.append(tc)
                                tool_id = block.get("id")
                                if tool_id:
                                    pending_by_id[tool_id] = tc

        return Trajectory(
            user_prompt=user_prompt,
            user_followup=" ".join(user_followups),
            tool_calls=tool_calls,
            assistant_turns=assistant_turns,
        )

    def format_context(self, hits: list[RetrievedSkill]) -> str:
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

    def is_shell_escape(self, prompt: str) -> bool:
        s = prompt.strip()
        if not s:
            return True
        if s.startswith(_BANG_PREFIXES):
            return True
        if "\n" not in s and len(s) < 120:
            first_token = s.split(None, 1)[0].lower()
            if first_token in _SHELL_CMD_HEADS:
                return True
        return False


def _ensure_hook(settings: dict[str, Any], event: str, command: str) -> bool:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise RuntimeError(".claude/settings.json 'hooks' must be an object")

    event_hooks = hooks.get(event)
    if event_hooks is None:
        event_hooks = []
        hooks[event] = event_hooks

    if not isinstance(event_hooks, list):
        raise RuntimeError(f".claude/settings.json hooks.{event} must be a list")

    for group in event_hooks:
        if not isinstance(group, dict):
            continue
        for hook in group.get("hooks", []) or []:
            if isinstance(hook, dict) and hook.get("command") == command:
                return False

    event_hooks.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": command}],
        }
    )
    return True


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


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
