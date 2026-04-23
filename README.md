<p align="center">
  <img src="docs/assets/hero.png" alt="Pixel art of a vintage CRT monitor with a tiny brain character flexing a bicep on its amber phosphor screen" width="720">
</p>

<h1 align="center">muscle-memory</h1>

<p align="center">
  <em>Procedural memory for coding agents. Your past sessions, compiled.</em>
</p>


`muscle-memory` gives coding agents and harnesses a memory that actually compounds. Instead of dumping prose into `CLAUDE.md` files that bloat every context, it watches sessions, extracts reusable **Skills** — executable playbooks with activation conditions, steps, and termination criteria — and retrieves the right ones on demand when you start a new task. Claude Code has the deepest runtime integration today, and Codex is also supported for setup, explicit retrieval, and offline transcript ingest.

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

Skills are **on-demand** (not always in context), **execution-scored** (good ones survive, bad ones die), **self-improving** (failing skills get rewritten via semantic-gradient PPO), and **user-editable** (plain text, no opaque embeddings).

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

# or: explicit harness selection for scripts/CI
mm init --harness claude-code
mm init --harness codex

# generic harness mode for other agents/orchestrators
mm init --harness generic

# optional: bootstrap from recent Claude Code history
mm bootstrap --days 30

# explicit retrieval for any harness
mm retrieve "run the tests in this repo" --json

# offline ingest of a Claude transcript
mm ingest transcript ./session.jsonl --format claude-jsonl --no-extract

# inspect what you've learned
mm list
mm show <skill-id>
mm stats
mm doctor
mm jobs list

# review quarantined candidates before they become retrievable
mm review list
mm review approve <skill-id>
mm review reject <skill-id>

# self-improvement
mm refine <skill-id>       # rewrite a skill via semantic-gradient PPO
mm refine --auto           # sweep all skills meeting auto-refine criteria
mm refine <id> --rollback  # undo the most recent refinement

# maintenance
mm maint pause         # pause hooks while you investigate a bad state
mm maint resume        # resume hooks after recovery
mm doctor              # inspect runtime health, retrieval decisions, and jobs
mm review list         # inspect quarantined candidates before promoting them
mm jobs retry-failed   # retry failed background extraction/refinement work
mm maint dedup             # collapse near-duplicate skills
mm maint rescore           # re-run the outcome heuristic on stored episodes
mm maint prune             # delete demonstrably bad skills
mm maint govern            # eval-driven demotion / review recommendations
```

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
- `mm ingest transcript ./codex-task.jsonl --format codex-jsonl --prompt "..."`

For other harnesses, initialize with `mm init --harness generic` and use:
- `mm retrieve ...` for explicit retrieval before prompting your agent
- `mm ingest transcript ...` or `mm ingest episode ...` for offline learning

For extraction/refinement, the default LLM backend is still `claude-code`. If you prefer API-key-based extraction, set `MM_LLM_PROVIDER=openai` and `OPENAI_API_KEY=***`.

## How it works

The core engine is harness-agnostic: retrieval, scoring, extraction, and storage live in the same memory layer regardless of runtime. The diagram below shows the Claude Code adapter path; other harnesses can use explicit `mm retrieve` plus offline `mm ingest ...`.

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
