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

The deepest runtime surface is still the Claude Code harness, and the
behavioral release proof is still Claude-specific. Codex setup and ingest
flows are also supported in the fast suites. If a session gets into a bad
state, the same recovery commands used in production docs apply here:
`mm maint pause`, `mm maint resume`, `mm doctor`, `mm review list`, and
`mm jobs retry-failed`.

To verify the first release gate locally, run the release preflight:

```bash
uv run python scripts/release_preflight.py $(uv run python - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
print(pyproject['project']['version'])
PY
)
```

For the full CI-equivalent release path, also run the benchmark gate, build,
distribution checks, checksum generation, wheel/sdist smoke tests, and the
current-platform binary build/smoke path. The complete ordered release
sequence lives in [docs/release.md](release.md).

If you want the lower-level package checks individually:

```bash
uv run python scripts/release_benchmark_gate.py > benchmark-run.json
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

If you are exercising the curl-install release path locally, also build and
smoke-test the standalone binary for your current target:

```bash
uv sync --extra dev --extra openai --extra voyage
uv run python scripts/build_release_binaries.py 0.12.0 dist --target darwin-arm64
uv run python scripts/check_release_binaries.py 0.12.0 dist --target darwin-arm64
```

On Linux, swap the target to `linux-x86_64`. The GitHub release job then merges
those binaries with `scripts/install.sh` and a `SHA256SUMS` manifest before
publishing the release payload. Package artifact attestations still cover the
wheel and sdist path.

## Running the CLI during development

Two options:

```bash
# option 1: via the editable venv (after ./scripts/dev-sync)
.venv/bin/mm list
.venv/bin/mm init
.venv/bin/mm init --harness codex

# option 2: via PYTHONPATH
PYTHONPATH=src python -m muscle_memory list
PYTHONPATH=src python -m muscle_memory retrieve "run the tests" --json
```

## Publishing a release

1. Update the version in `pyproject.toml` and `src/muscle_memory/__init__.py`.
2. Add a matching `## [x.y.z]` section to `CHANGELOG.md`.
3. In GitHub Actions, run the `Release` workflow and pass that same version.

The workflow validates version consistency, runs the test suite, builds the
wheel and sdist, smoke-tests clean installs from both artifacts, builds the
macOS and Linux standalone binaries, generates GitHub artifact attestations for
the built distributions when the repository supports them, extracts the
matching changelog section as release notes, pushes tag `vX.Y.Z`, publishes the
GitHub release asset bundle (`install.sh`, binaries, checksums, wheel, and
sdist), and can publish to PyPI afterward as an optional non-blocking follow-up.

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
├── llm.py            # pluggable LLM backend for extraction/refinement
├── ingest.py         # offline transcript / episode ingest pipeline
├── extractor.py      # trajectory → candidate Skills
├── outcomes.py       # heuristic success/failure detection
├── retriever.py      # query → top-k Skills (hooks or explicit retrieve)
├── scorer.py         # credit assignment + pruning
├── bootstrap.py      # seed from Claude Code session history
├── cli.py            # `mm` typer app
├── harness/
│   ├── base.py       # harness adapter interface
│   ├── claude_code.py# Claude Code runtime adapter
│   ├── codex.py      # Codex transcript + setup adapter
│   └── generic.py    # harness-agnostic / offline-only adapter
├── hooks/
│   ├── user_prompt.py   # UserPromptSubmit handler
│   ├── stop.py          # Stop handler + transcript parser
│   └── install.py       # routes init/install through harness adapters
└── prompts/
    └── extract.md    # skill extraction prompt (version-controlled)
```
