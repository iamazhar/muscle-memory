"""Pluggable LLM backends for extraction and rerank.

Default is Anthropic Claude Haiku 4.5 — cheap, fast, and good enough
for structured skill extraction from trajectories.
"""

from __future__ import annotations

import json
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
    ) -> str:
        ...


class AnthropicLLM:
    """Anthropic Claude client for structured extraction tasks."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ):
        try:
            from anthropic import Anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "anthropic not installed. Run: uv add anthropic"
            ) from exc
        self.model = model
        self._api_key = api_key
        self._client: object | None = None

    def _load(self) -> None:
        if self._client is None:
            from anthropic import Anthropic

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = Anthropic(**kwargs)

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
        resp = self._client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        """Ask for JSON, parse it, be forgiving about code fences."""
        text = self.complete_text(
            system + "\n\nRespond with ONLY valid JSON. No prose, no code fences.",
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _extract_json(text)


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


def make_llm(config: Config) -> LLM:
    provider = config.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(model=config.llm_model, api_key=config.llm_api_key)
    raise ValueError(f"Unknown LLM provider: {provider}")
