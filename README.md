<p align="center">
  <img src="docs/assets/hero.png" alt="Pixel art of a vintage CRT monitor with a tiny brain character flexing a bicep on its amber phosphor screen" width="720">
</p>

<h1 align="center">muscle-memory</h1>

<p align="center">
  <em>Procedural memory for coding agents. Your past sessions, compiled.</em>
</p>


`muscle-memory` gives Claude Code a memory that actually compounds. Instead of dumping prose into `CLAUDE.md` files that bloat every context, it watches your sessions, extracts reusable **Skills** — executable playbooks with activation conditions, steps, and termination criteria — and retrieves the right ones on demand when you start a new task.

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
# install (Anthropic default)
uv tool install muscle-memory

# or with OpenAI support baked in
uv tool install 'muscle-memory[openai]'

# in your project
cd ~/code/my-project
mm init                    # creates .claude/mm.db, registers Claude Code hooks

# optional: bootstrap from recent history
mm bootstrap --days 30

# now just use Claude Code normally.
# skills accumulate automatically.

# inspect what you've learned
mm list
mm show <skill-id>
mm stats

# review quarantined candidates before they become retrievable
mm review list
mm review approve <skill-id>
mm review reject <skill-id>

# self-improvement
mm refine <skill-id>       # rewrite a skill via semantic-gradient PPO
mm refine --auto           # sweep all skills meeting auto-refine criteria
mm refine <id> --rollback  # undo the most recent refinement

# maintenance
mm maint dedup             # collapse near-duplicate skills
mm maint rescore           # re-run the outcome heuristic on stored episodes
mm maint prune             # delete demonstrably bad skills
```

## Try The Demo

If you want a realistic place to dogfood the product immediately, use the in-repo
[OrbitOps demo](demo/orbitops/README.md). It is a tiny fictional SaaS app with a
marketing page, an interactive dashboard, a local smoke-check command, and its own
project-local `.claude` anchor so skills stay isolated from the repo root.

## Authentication

No API key needed. Extraction shells out to `claude -p`, so it uses your
existing Claude Code subscription auth. Just be logged into Claude Code.

For OpenAI as an alternative backend: `export MM_LLM_PROVIDER=openai`
and `export OPENAI_API_KEY=sk-...`.

## How it works

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

Candidates can be promoted automatically when the same procedure is learned from
multiple distinct successful episodes, or manually via `mm review approve`.

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
