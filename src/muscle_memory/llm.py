"""Pluggable LLM backends for extraction and refinement.

Default is claude-code, which shells out to `claude -p` and uses
the user's existing Claude Code subscription auth. Zero additional
API keys or billing needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Protocol, runtime_checkable

from muscle_memory.config import Config


@runtime_checkable
class LLM(Protocol):
    """Minimal interface for an LLM that can return structured JSON."""

    model: str

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        """Return a parsed JSON object / list produced by the LLM."""
        ...

    def complete_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str: ...


def _extract_json(text: str) -> Any:
    """Strip code fences, find first JSON value, parse it.

    LLMs often drop JSON inside ```json ... ``` fences even when asked
    not to. We're tolerant.
    """
    s = text.strip()
    if s.startswith("```"):
        # strip first fence line and trailing fence
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # find the first { or [
    for i, ch in enumerate(s):
        if ch in "{[":
            s = s[i:]
            break

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # last-ditch: try to find a balanced JSON array/object
        return json.loads(_find_balanced(s))


def _find_balanced(s: str) -> str:
    """Return the first balanced JSON value in s."""
    if not s:
        raise ValueError("empty")
    opener = s[0]
    closer = {"{": "}", "[": "]"}.get(opener)
    if closer is None:
        raise ValueError(f"unexpected first char: {opener!r}")
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    raise ValueError("unbalanced JSON")


class OpenAILLM:
    """OpenAI client for structured extraction. Uses the Chat Completions API
    with `response_format={'type': 'json_object'}` for reliable JSON output.

    Install with: `uv tool install 'muscle-memory[openai]'` or just
    `pip install openai` in your env. Reads OPENAI_API_KEY from env by
    default; override via `MM_LLM_API_KEY`.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. "
                "Run: uv tool install 'muscle-memory[openai]' "
                "or: pip install openai"
            ) from exc
        self.model = model
        self._api_key = api_key
        self._client: object | None = None

    def _load(self) -> None:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)

    def complete_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        self._load()
        assert self._client is not None
        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        """OpenAI's JSON mode requires a top-level object, but our
        extractor expects a list. We wrap the instruction so the model
        returns `{"skills": [...]}` and we unwrap it here.
        """
        self._load()
        assert self._client is not None

        wrapped_system = (
            system + '\n\nReturn a JSON object with a single key "skills" whose '
            "value is the JSON array described above. "
            'Example: {"skills": [ ... ]}'
        )

        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": wrapped_system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or "{}"
        data = _extract_json(text)
        if isinstance(data, dict) and "skills" in data:
            return data["skills"]
        # fall back to whatever came back
        return data


class ClaudeCodeLLM:
    """LLM backend that shells out to `claude -p`.

    Uses the user's existing Claude Code subscription auth, so no
    API key or billing credits are needed. Runs in --bare mode to
    skip hooks, LSP, and other overhead.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._claude_path: str | None = shutil.which("claude")

    def _run(self, system: str, user: str, *, max_tokens: int = 2048) -> str:
        if self._claude_path is None:
            raise RuntimeError(
                "claude CLI not found on PATH. "
                "Install Claude Code or use MM_LLM_PROVIDER=anthropic with an API key."
            )
        result = subprocess.run(
            [
                self._claude_path,
                "-p",
                "--bare",
                "--system-prompt",
                system,
                "--output-format",
                "json",
                "--max-turns",
                "1",
                "--model",
                self.model,
            ],
            input=user,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        # Parse the JSON output to extract the result text
        try:
            events = json.loads(result.stdout)
            if isinstance(events, list):
                for event in reversed(events):
                    if isinstance(event, dict) and event.get("type") == "result":
                        return str(event.get("result", ""))
            return result.stdout
        except json.JSONDecodeError:
            return result.stdout

    def complete_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        return self._run(system, user, max_tokens=max_tokens)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        text = self._run(
            system + "\n\nRespond with ONLY valid JSON. No prose, no code fences.",
            user,
            max_tokens=max_tokens,
        )
        return _extract_json(text)


def make_llm(config: Config) -> LLM:
    provider = config.llm_provider.lower()
    if provider == "claude-code":
        return ClaudeCodeLLM(model=config.llm_model)
    if provider == "openai":
        model = config.llm_model
        if model.startswith("claude"):
            model = "gpt-4o-mini"
        return OpenAILLM(model=model, api_key=config.llm_api_key)
    raise ValueError(f"Unknown LLM provider: {provider}")
