"""Extract one version section from CHANGELOG.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_release_notes(version: str, changelog_text: str) -> str:
    heading = f"## [{version}]"
    if heading not in changelog_text:
        raise ValueError(f"CHANGELOG.md does not contain {heading}")

    lines = changelog_text.splitlines()
    collecting = False
    out: list[str] = []

    for line in lines:
        if line.startswith("## ["):
            if line.startswith(heading):
                collecting = True
                out.append(line)
                continue
            if collecting:
                break
        elif collecting:
            out.append(line)

    notes = "\n".join(out).strip()
    if not notes:
        raise ValueError(f"No release notes found for {version}")
    return notes + "\n"


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: release_notes.py <version>", file=sys.stderr)
        return 1

    version = args[0].strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version):
        print(f"invalid version: {version!r}", file=sys.stderr)
        return 1

    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    try:
        sys.stdout.write(extract_release_notes(version, changelog))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
