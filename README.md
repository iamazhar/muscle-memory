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
Session 2:  "run the tests" → Skill activates automatically, zero rediscovery
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

# self-improvement
mm refine <skill-id>       # rewrite a skill via semantic-gradient PPO
mm refine --auto           # sweep all skills meeting auto-refine criteria
mm refine <id> --rollback  # undo the most recent refinement

# maintenance
mm dedup                   # collapse near-duplicate skills
mm rescore                 # re-run the outcome heuristic on stored episodes
mm prune                   # delete demonstrably bad skills
```

## Authentication

`muscle-memory` needs an LLM for skill extraction (runs after each session,
not on every prompt). **It cannot reuse your Claude Code subscription auth**
— that's a known limitation; Anthropic does not currently expose a
subscription-capable SDK for third-party tools.

Your options:

| Provider | Setup | Cost per session (measured) |
|---|---|---|
| **Anthropic** (default) | `export ANTHROPIC_API_KEY=sk-ant-...` — **needs API credits, not a Max/Pro subscription** | **~$0.009** (Sonnet 4.6, extraction only; refinement is +$0.030 when it fires) |
| **OpenAI** | `uv tool install 'muscle-memory[openai]'` then `export OPENAI_API_KEY=sk-...` and `export MM_LLM_PROVIDER=openai` | lower but unmeasured — `gpt-4o-mini` default |
| **Local / Ollama** | *(planned, not yet implemented)* | free |

At 20 sessions per day, Anthropic's default comes out to ~$5.50/month for a heavy user. See [`docs/performance.md`](docs/performance.md) for the full measurement methodology.

If you already use Claude Code via a Max/Pro subscription, you'll still
need a separate Anthropic API key with billing credits, or use OpenAI.
See [docs/authentication.md](docs/authentication.md) for details.

## How it works

```
┌────────────────────────── Claude Code Session ────────────────────────────┐
│                                                                            │
│  user prompt ──────►  ┌───────────────┐      ┌─────────────────────┐       │
│                       │  Retriever    │─────►│  inject playbook    │       │
│                       │  (fastembed + │      │  with 🧠 marker +   │       │
│                       │   sqlite-vec) │      │  imperative framing │       │
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
│     │  Extractor        │       │     SQLite + sqlite-vec     │           │
│     │  (async, on new   │──────►│     (.claude/mm.db)         │           │
│     │   trajectories)   │       └─────────────────────────────┘           │
│     └───────────────────┘                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

Inspired directly by [ProcMEM (arxiv:2602.01869)](https://arxiv.org/abs/2602.01869). The full three-stage refinement loop — semantic gradient → LLM rewrite → PPO-Gate trust-region verification — is adapted for Anthropic's API (which doesn't expose token logprobs, so the PPO Gate uses an LLM-judge proxy over stored trajectories). See the [CHANGELOG](CHANGELOG.md) for the full story.

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

## Status

**v0.2.0 — alpha, live on PyPI, install with `uv tool install muscle-memory`.**

What works today:

- Conservative LLM-driven extraction with a version-controlled prompt
- Fast embedding-based retrieval (<500ms target, 2000-skill tests flat at ~3.5ms)
- Visible 🧠 marker + imperative execution framing
- Heuristic outcome inference for sessions without explicit rewards
- Darwinian skill scoring with maturity tiers (candidate → established → proven)
- Dual-layer dedup: insertion gate + on-demand cluster consolidation
- **Full Non-Parametric PPO refinement loop** — semantic gradient → LLM rewrite → PPO-Gate verification
- 121 passing unit + integration + edge-case tests, 8 opt-in behavioral tests against real Claude Code

Planned / open:

- Persistent embedder daemon for sub-100ms seeded-store hook latency
- Local Ollama backend for zero-API-cost extraction
- Cross-model skill transfer and Q-value retrieval ranking (paper's open items)
- Windows support (currently untested)

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — full version history
- [docs/authentication.md](docs/authentication.md) — detailed auth + provider setup
- [docs/performance.md](docs/performance.md) — measured latency + cost numbers, deferred optimizations
- [docs/testing.md](docs/testing.md) — test layers + the `claude -p` gotcha
- [docs/development.md](docs/development.md) — contributor setup, including the macOS uv `.pth` hidden-flag workaround

## License

MIT — see [LICENSE](LICENSE).
