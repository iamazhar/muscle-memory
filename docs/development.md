# Development

## Setup

```bash
./scripts/dev-sync
```

This runs `uv sync --extra dev` and then clears the macOS hidden-file flag
that uv sets on its editable `.pth` files. (See **known gotcha** below.)

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Or, if you want the installed-tool experience end-to-end:

```bash
uv tool uninstall muscle-memory 2>/dev/null || true
uv cache clean muscle-memory 2>/dev/null || true
uv tool install --from . muscle-memory
mm --version
```

To verify release artifacts exactly the way CI does, run the release preflight:

```bash
uv run python scripts/release_preflight.py $(uv run python - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
print(pyproject['project']['version'])
PY
)
```

If you want the lower-level steps individually:

```bash
uv build
uv run python scripts/generate_release_checksums.py $(uv run python - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
print(pyproject['project']['version'])
PY
)
uv run python scripts/check_release_artifacts.py $(uv run python - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
print(pyproject['project']['version'])
PY
)
```

## Running the CLI during development

Two options:

```bash
# option 1: via the editable venv (after ./scripts/dev-sync)
.venv/bin/mm list

# option 2: via PYTHONPATH
PYTHONPATH=src python -m muscle_memory list
```

## Publishing a release

1. Update the version in `pyproject.toml` and `src/muscle_memory/__init__.py`.
2. Add a matching `## [x.y.z]` section to `CHANGELOG.md`.
3. In GitHub Actions, run the `Release` workflow and pass that same version.

The workflow validates version consistency, runs the test suite, builds the
wheel and sdist, smoke-tests clean installs from both artifacts, generates
GitHub artifact attestations for the built distributions, extracts the matching
changelog section as release notes, pushes tag `vX.Y.Z`, can publish to PyPI,
emits a `SHA256SUMS` manifest for the release artifacts, and creates a GitHub
release.

If you also maintain the separate Homebrew tap, bump the formula there after
the GitHub release is live so `brew update` can discover the new version.

## Known gotcha: macOS + uv + Python 3.12 hidden `.pth` files

uv sets the macOS `UF_HIDDEN` flag on the `.pth` files it drops into
`.venv/lib/python*/site-packages/`. Starting in Python 3.12, `site.py`
respects that flag and skips hidden `.pth` files entirely. The symptom
is that `python -c "import muscle_memory"` fails with
`ModuleNotFoundError` even though the source is right there.

Workarounds:

1. **`./scripts/dev-sync`** — syncs then unhides. Preferred.
2. **`chflags nohidden .venv/lib/python*/site-packages/*.pth`** — manual.
3. **`PYTHONPATH=src`** — skips `.pth` processing entirely.

Run a tracking issue upstream if uv's behaviour ever changes.

## Project layout

```
src/muscle_memory/
├── models.py         # pydantic Skill / Episode / Trajectory
├── db.py             # SQLite + sqlite-vec DAO
├── config.py         # env var + path resolution
├── embeddings.py     # pluggable Embedder (fastembed, openai, voyage)
├── llm.py            # pluggable LLM (anthropic default)
├── extractor.py      # trajectory → candidate Skills
├── outcomes.py       # heuristic success/failure detection
├── retriever.py      # query → top-k Skills (fast path for hooks)
├── scorer.py         # credit assignment + pruning
├── bootstrap.py      # seed from Claude Code session history
├── cli.py            # `mm` typer app
├── hooks/
│   ├── user_prompt.py   # UserPromptSubmit handler
│   ├── stop.py          # Stop handler + transcript parser
│   └── install.py       # wires .claude/settings.json
└── prompts/
    └── extract.md    # skill extraction prompt (version-controlled)
```
