# OrbitOps Demo

`OrbitOps` is a tiny fictional SaaS product inside this repo. It exists so people can
dogfood `muscle-memory` on believable frontend and product-surface work without needing
to bring their own project first.

## What is here

- a marketing page at `/`
- an interactive dashboard at `/dashboard`
- a local JSON endpoint at `/api/dashboard.json`
- a smoke-check command: `python check.py`

## Quickstart

```bash
cd demo/orbitops
python3 app.py
```

Open `http://127.0.0.1:8008`.

In another terminal, initialize `muscle-memory` against this demo project:

```bash
cd demo/orbitops
mm init --scope project
```

If you are developing from the repo without installing the tool globally, this works too:

```bash
cd demo/orbitops
PYTHONPATH=../../src ../../.venv/bin/python -m muscle_memory init --scope project
```

The committed `.claude/` directory makes OrbitOps behave like its own project root, so
the demo writes to `demo/orbitops/.claude/mm.db` instead of the repo-root memory store.

## Local verification

```bash
cd demo/orbitops
python3 check.py
```

## Dogfood prompts

See [DOGFOOD.md](DOGFOOD.md) for suggested tasks that should generate useful repeated
workflows while you build on top of the demo.
