# muscle-memory

> Procedural memory for coding agents. Your past sessions, compiled.

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
Session 5:  Skill gets refined from more examples
Session 20: Skill has been invoked 18x, 17 successes → promoted to "proven"
Session 50: Unused skills auto-pruned, active ones keep improving
```

Skills are **on-demand** (not always in context), **execution-scored** (good ones survive, bad ones die), and **user-editable** (plain text, no opaque embeddings).

## Quickstart

```bash
# install
uv tool install muscle-memory

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
```

## How it works

```
┌──────────────────────────────────────────────────────┐
│                 Claude Code Session                   │
│                                                        │
│  user prompt ─┐                                       │
│               ▼                                        │
│       ┌───────────────┐     top skills                │
│       │  Retriever    │────────────►  inject context  │
│       │  (embedding)  │                                │
│       └───────────────┘                                │
│               │                                        │
│               ▼                                        │
│       [ LLM executes with Skill hints ]               │
│               │                                        │
│               ▼                                        │
│       ┌───────────────┐                                │
│       │   Extractor   │  proposes new Skills           │
│       │   (async)     │                                │
│       └───────┬───────┘                                │
│               │                                        │
│               ▼                                        │
│       ┌────────────────────┐                          │
│       │   Scorer           │  updates / prunes        │
│       └────────────────────┘                          │
│               │                                        │
│               ▼                                        │
│       ┌────────────────────┐                          │
│       │   SQLite + vec     │                          │
│       └────────────────────┘                          │
└──────────────────────────────────────────────────────┘
```

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

**Alpha.** Core extraction + retrieval + scoring are working. The Non-Parametric PPO refinement loop from the paper is planned for v2.

## License

MIT — see [LICENSE](LICENSE).
