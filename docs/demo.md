# Demo App

`muscle-memory` now ships with a small in-repo demo application: [OrbitOps](../demo/orbitops/README.md).

The goal is simple: give engineers a realistic surface where the product can be
dogfooded immediately.

This demo is part of the supported dogfood surface. Claude Code still has the
deepest runtime integration here, and Codex can use the same project setup plus
explicit retrieval and transcript learning. When a dogfood session drifts, start
with `mm status` and `mm doctor`; lower-level recovery commands are available
under the advanced command groups.

## Why it exists

- new users can see the product on something concrete instead of a synthetic transcript
- contributors have a stable place to repeat workflows and observe what skills emerge
- retrieval quality is easier to judge when the same app is edited across multiple sessions

## Quickstart

```bash
cd demo/orbitops
python3 app.py
```

Then initialize `muscle-memory` against the demo app.

If you are using the interactive chooser in a terminal:

```bash
cd demo/orbitops
mm init --scope project
```

If you are using Claude Code hooks directly:

```bash
cd demo/orbitops
mm init --scope project --harness claude-code
```

If you want the Codex or harness-agnostic path instead:

```bash
cd demo/orbitops
mm init --scope project --harness codex
mm retrieve "tighten the hero copy and rerun checks" --json

cd demo/orbitops
mm init --scope project --harness generic
mm retrieve "tighten the hero copy and rerun checks" --json
```

If you are developing from this repo without a global install:

```bash
cd demo/orbitops
PYTHONPATH=../../src ../../.venv/bin/python -m muscle_memory init --scope project
```

The committed `.claude/` directory makes the demo behave like its own project root,
so the memory database lands in `demo/orbitops/.claude/mm.db`.

## Recommended dogfood loop

1. Start the local app with `python3 app.py`.
2. Ask Claude to make a small change to the landing page or dashboard.
3. Run `python3 check.py`.
4. Verify both `/` and `/dashboard`.
5. Repeat a few times and inspect `mm status` plus `mm skills`.

See [demo/orbitops/DOGFOOD.md](../demo/orbitops/DOGFOOD.md) for prompt ideas.
