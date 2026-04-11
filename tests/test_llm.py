"""Tests for LLM backend helpers."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from muscle_memory.llm import ClaudeCodeLLM, _extract_claude_result


def test_extract_claude_result_reads_result_event() -> None:
    stdout = json.dumps(
        [
            {"type": "system", "subtype": "init"},
            {"type": "result", "result": '{"ok": true}', "is_error": False},
        ]
    )
    text, is_error = _extract_claude_result(stdout)
    assert text == '{"ok": true}'
    assert is_error is False


def test_claude_code_llm_surfaces_cli_result_text_on_error() -> None:
    stdout = json.dumps(
        [
            {
                "type": "result",
                "result": "Not logged in · Please run /login",
                "is_error": True,
            }
        ]
    )

    with patch("shutil.which", return_value="claude"):
        llm = ClaudeCodeLLM()

    completed = subprocess.CompletedProcess(
        args=["claude"],
        returncode=1,
        stdout=stdout,
        stderr="",
    )
    with patch("subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="Not logged in"):
            llm.complete_text("system", "user")

