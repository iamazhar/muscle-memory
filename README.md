<p align="center">
  <img src="docs/assets/hero.png" alt="Pixel art of a vintage CRT monitor with a tiny brain character flexing a bicep on its amber phosphor screen" width="720">
</p>

<h1 align="center">muscle-memory</h1>

<p align="center">
  <em>Practiced skill for coding agents. Better outcomes with fewer tokens.</em>
</p>


`muscle-memory` gives coding agents practiced skill. It turns past successful work into compact, reusable **Skills**: executable playbooks with activation conditions, steps, and termination criteria. The goal is simple: better outcomes, less rediscovery, and fewer tokens than an agent that starts from scratch every session.

Instead of dumping prose into `CLAUDE.md` files that bloat every context, `muscle-memory` retrieves only the relevant playbook when the task calls for it. Claude Code has the deepest runtime integration today, and Codex is supported for setup, explicit retrieval, and offline transcript ingest.

Inspired by [ProcMEM (arxiv:2602.01869)](https://arxiv.org/abs/2602.01869), but purpose-built for coding agents.

## The problem

```
Session 1:  Figure out the monorepo's weird test runner (15 min)
Session 2:  Figure it out again (15 min)
Session 3:  Add a note to CLAUDE.md (2 min, bloats future context)
Session 50: CLAUDE.md is 4000 lines, half stale, nobody reads it
```

## What muscle-memory does instead

```
Session 1:  Figure out the test runner (15 min) → extractor creates a Skill
Session 2:  extractor stores it as a candidate, not yet trusted
Session 3:  same pattern shows up again → promoted to "live"
Session 4:  "run the tests" → Skill activates automatically, zero rediscovery
Session 5:  Skill execution had a subtle bug → PPO refiner rewrites just that step
Session 20: Skill has been invoked 18x, 17 successes → promoted to "proven"
Session 50: Unused skills auto-pruned, active ones keep improving
```

Skills are **on-demand** (not always in context), **execution-scored** (good ones survive, bad ones die), and **user-editable** (plain text, no opaque embeddings).

### You'll see it working

When a skill fires, Claude's response is prefixed with a visible marker so you always know when muscle-memory is doing something:

```
🧠 muscle-memory: executing playbook — After uv sync on macOS…
```

Then Claude **executes the playbook directly** — runs the commands, makes the edits, verifies the result — instead of narrating the steps back at you. If no skill matches, Claude proceeds silently with no marker.

## Quickstart

```bash
# install from GitHub Releases (recommended)
curl -fsSL https://github.com/iamazhar/muscle-memory/releases/latest/download/install.sh | sh
mm --version
```

The GitHub Releases installer downloads the matching standalone binary for your
platform, verifies it against `SHA256SUMS`, and installs `mm` into
`${MM_INSTALL_DIR:-$HOME/.local/bin}`.

To pin a specific release:

```bash
curl -fsSL https://github.com/iamazhar/muscle-memory/releases/latest/download/install.sh | MM_VERSION=0.11.0 sh
```

If you specifically want the Python package from a tagged release instead of the
standalone GitHub Releases binary, install from the release tag directly:

```bash
# replace v0.11.0 with the release tag you want
uv tool install "muscle-memory @ git+https://github.com/iamazhar/muscle-memory.git@v0.11.0"

# or with OpenAI support baked in for extraction/refinement
uv tool install "muscle-memory[openai] @ git+https://github.com/iamazhar/muscle-memory.git@v0.11.0"
```

Once installed, initialize muscle-memory inside your project:

```bash
cd ~/code/my-project

# choose Claude Code or Codex in your terminal
mm init

# explicit harness selection for scripts/CI
mm init --harness claude-code
mm init --harness codex

# learn from recent Claude Code history
mm learn --days 30

# retrieve practiced skill for any harness
mm retrieve "run the tests in this repo" --json

# learn from an explicit transcript
mm learn --transcript ./session.jsonl --format claude-jsonl --no-extract

# inspect outcomes, token efficiency, and learned skills
mm status
mm skills
mm show <skill-id>
mm doctor
```

The primary loop is intentionally small:

- `mm init` connects a project to a harness.
- `mm learn` turns session history or transcripts into reusable skill candidates.
- `mm retrieve` brings back the relevant skill for the current task.
- `mm status` shows whether the store is producing reuse, successful outcomes, and token savings.
- `mm skills` and `mm show` let you inspect the compact playbooks directly.

Advanced operator commands still exist for repair, review, jobs, evaluation, and imports, but they are not part of the normal workflow.

## Try The Demo

If you want a realistic place to dogfood the product immediately, use the in-repo
[OrbitOps demo](demo/orbitops/README.md). It is a tiny fictional SaaS app with a
marketing page, an interactive dashboard, a local smoke-check command, and its own
project-local `.claude` anchor so skills stay isolated from the repo root.

## Runtime adapters and authentication

Run `mm init` in a real terminal to choose between Claude Code and Codex. The selected harness is saved in `.claude/mm.json` for future project-local commands. In scripts or CI, pass `--harness` explicitly.

For Claude Code runtime integration, `mm init --harness claude-code` installs hooks into `.claude/settings.json` and uses the existing Claude Code session to capture prompts/transcripts.

For Codex, `mm init --harness codex` initializes the project database and saved harness config, but does not install automatic prompt hooks yet. Use:
- `mm retrieve ...` for explicit retrieval before prompting Codex
- `mm learn --transcript ./codex-task.jsonl --format codex-jsonl --prompt "..."`

For other harnesses, initialize with `mm init --harness generic` and use:
- `mm retrieve ...` for explicit retrieval before prompting your agent
- `mm learn --transcript ...` for offline learning

For extraction/refinement, the default LLM backend is still `claude-code`. If you prefer API-key-based extraction, set `MM_LLM_PROVIDER=openai` and `OPENAI_API_KEY=***`.

## How it works

The core engine is harness-agnostic: retrieval, scoring, extraction, and storage live in the same memory layer regardless of runtime. The diagram below shows the Claude Code adapter path; other harnesses can use explicit `mm retrieve` plus offline `mm learn --transcript ...`.

```
┌────────────────────────── Claude Code Session ────────────────────────────┐
│                                                                            │
│  user prompt ──────►  ┌───────────────┐      ┌─────────────────────┐       │
│                       │  Retriever    │─────►│  inject playbook    │       │
│                       │  (fastembed + │      │  with 🧠 marker +   │       │
│                       │   sqlite)     │      │  imperative framing │       │
│                       └───────────────┘      └─────────┬───────────┘       │
│                                                         │                   │
│                                                         ▼                   │
│                        [ Claude executes the playbook — runs Bash, edits, ] │
│                        [ verifies. Not narrated back at the user. ]         │
│                                                         │                   │
│           turn end ─────────────────────────────────────┤                   │
│                                                         ▼                   │
│  ┌──────────────┐   ┌────────────────┐   ┌────────────────────────────┐    │
│  │  Stop hook   │──►│    Scorer      │──►│  PPO Refiner (async)       │    │
│  │  parses      │   │  credits +     │   │  if a skill is failing:    │    │
│  │  transcript  │   │  prunes        │   │  1. semantic gradient      │    │
│  │  + infers    │   │                │   │  2. LLM rewrite skill text │    │
│  │  outcome     │   └────────┬───────┘   │  3. PPO-Gate verification  │    │
│  └──────┬───────┘            │           └────────────┬───────────────┘    │
│         │                    │                        │                    │
│         └───┬────────────────┴────────────────────────┘                    │
│             ▼                                                              │
│     ┌───────────────────┐       ┌─────────────────────────────┐           │
│     │  Extractor        │       │     SQLite                  │           │
│     │  (async, on new   │──────►│     (.claude/mm.db)         │           │
│     │   trajectories)   │       └─────────────────────────────┘           │
│     └───────────────────┘                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

Inspired directly by [ProcMEM (arxiv:2602.01869)](https://arxiv.org/abs/2602.01869). The full three-stage refinement loop (semantic gradient, LLM rewrite, PPO-Gate trust-region verification) uses an LLM-judge proxy over stored trajectories. See the [CHANGELOG](CHANGELOG.md) for the full story.

## Skill lifecycle

Newly extracted skills do not go straight into live retrieval.

- `candidate`: quarantined; stored for review or repeated evidence, but never auto-injected
- `live`: trusted enough to retrieve automatically
- `proven`: repeatedly successful and strongly trusted

Candidates are promoted conservatively: they need repeated evidence from distinct successful source episodes before auto-promotion, or manual approval via `mm review approve`. Trust can also move in the other direction now — weak `live`/`proven` skills can be flagged for review or demotion by the eval-governance pass.

## Skill anatomy

Each Skill is three editable text fields:

```json
{
  "activation":  "When pytest fails with ModuleNotFoundError in this monorepo",
  "execution":   "1. Check if tools/test-runner.sh exists.\n2. If yes, use it instead of invoking pytest directly.\n3. Set PYTEST_ADDOPTS=--no-cov for speed.",
  "termination": "Tests pass, or runner is confirmed missing",
  "tool_hints":  ["Bash: tools/test-runner.sh"]
}
```

No DSL. No code templates. Plain English that the agent reads and executes with judgment.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — full version history
- [docs/authentication.md](docs/authentication.md) — detailed auth + provider setup
- [docs/demo.md](docs/demo.md) — run the OrbitOps demo app and dogfood repeated workflows
- [docs/performance.md](docs/performance.md) — measured latency + cost numbers, deferred optimizations
- [docs/quality.md](docs/quality.md) — skill admission policy and anti-junk quality gates
- [docs/testing.md](docs/testing.md) — test layers + the `claude -p` gotcha
- [docs/development.md](docs/development.md) — contributor setup, including the macOS uv `.pth` hidden-flag workaround

## License

MIT — see [LICENSE](LICENSE).
