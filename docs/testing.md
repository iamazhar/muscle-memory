# Testing

## Test layers

`muscle-memory` is covered by three layers of tests:

| Suite | Count | Runtime | Requires |
|---|---|---|---|
| Unit (`test_models`, `test_db`, `test_outcomes`, `test_extractor`, `test_retriever`, `test_dedup`) | 49 | <100ms | Python deps only |
| Integration (`test_integration`) | 23 | <200ms | Python deps only |
| Edge cases (`test_edge_cases`) | 27 | ~500ms | Python deps (tests concurrency + 500-skill scale) |
| Behavioral (`test_behavioral`) | 8 | ~60s | `claude` + `mm` on PATH, **OPT-IN** |

Total: **99 passing tests** run by `pytest` in under a second. Behavioral
tests are skipped by default because they spawn real Claude Code sessions.

## Running tests

```bash
# after the dev-sync workaround for the macOS .pth hidden flag:
./scripts/dev-sync

# default: unit + integration + edge cases (fast)
.venv/bin/python -m pytest tests/

# include slow behavioral tests (real Claude Code)
CLAUDE_TESTS=1 .venv/bin/python -m pytest tests/test_behavioral.py -v
```

## The `claude -p` final-text-only gotcha

**You cannot verify the `🧠 muscle-memory:` marker by scanning `claude -p` stdout when the response involves tool calls.**

When Claude runs `UserPromptSubmit`-injected playbooks, the response
structure is typically:

1. Assistant text — `🧠 **muscle-memory**: executing playbook — …`
2. Tool use — `Bash(touch ...)`
3. Tool result
4. (more tool calls)
5. Assistant text — `"Fixed. The marker file has been created."`

`claude -p` returns only the **final** assistant message (step 5), so the
marker emitted in step 1 is gone from stdout. In interactive Claude Code
the user sees every turn, so the marker is visible — it's only the
scripting path that trims it.

### Workarounds for tests

Three strategies, in order of preference:

1. **Check observable side effects instead of stdout.** If the skill's
   execution includes `touch /path/to/marker`, check the file exists on
   disk. It can only be there if the Bash tool was actually invoked.
   `test_full_lifecycle_seed_execute_score` uses this strategy.

2. **Use `--output-format stream-json --verbose`** to capture every
   assistant event (not just the final text), then scan the JSONL for
   the marker. `test_matching_prompt_emits_brain_marker_in_stream` does
   this via the `_stream_contains_marker` helper.

3. **Design the test around short single-turn responses.** When
   Claude's reply is one text turn with no tool calls, the marker
   survives to `-p` stdout. `test_no_match_marker_survives_plain_p`
   exploits this — a "capital of France?" prompt produces a one-line
   answer with no marker at all (silent no-match).

### What this does NOT affect

- Real interactive Claude Code sessions. Users see the marker.
- The underlying mechanism. The marker is being emitted correctly; it's
  just being filtered out by `-p` mode's "return the final answer"
  semantics.
- Tool execution. Claude runs playbook steps in both modes.
- Scoring. Stop hook and scorer work identically in both modes.

## Retriever permissiveness + wrapper judgment

Behavioral testing surfaced that the retriever is intentionally
permissive — it returns top-k nearest skills without a hard semantic
gate. Distance cutoff is 1.5 (L2), which admits almost any match for
typical natural-language queries against the store.

The safety mechanism is the **injection wrapper**, which instructs
Claude:

> If a playbook's `Activate when` clearly does NOT fit the current task,
> ignore it and proceed normally.
>
> If NONE of the playbooks apply to the current task, instead start with:
> *(no marker emitted — Claude proceeds silently)*

This composite approach was verified in the agent-driven tests: when a
Kubernetes skill was the only skill in the store and the user asked
"what is the capital of France?", Claude correctly emitted the
no execution marker instead of fabricating a connection to kubectl. Layered safety — permissive retriever + smart wrapper +
Claude's own judgment — works.

If you want tighter retrieval, lower the distance cutoff in
`muscle_memory/retriever.py` or set a stricter
`retrieval_similarity_floor` in config. Trade-off: fewer false positives
at the cost of occasionally missing a real match with unusual wording.

## Hook failure is always silent

By design, both `mm hook user-prompt` and `mm hook stop` wrap their
entire bodies in `try/except Exception`, return exit 0, and print
nothing on any error. Rationale: **a hook crash must never break the
user's Claude Code session**.

This is tested in `test_edge_cases.py::TestHookResilience` with 9+
garbage-input cases (empty, binary, null, wrong-shape JSON, missing
fields, nonexistent transcript paths). Every one returns 0 cleanly.

If you're debugging why a skill isn't firing and suspect a silent
hook crash, temporarily set `MM_DEBUG=1` (not implemented yet — file
a feature request if you need it).

## Parallel behavioral testing

The behavioral test scenarios were designed to be runnable via
independent sub-agents spawned from a parent Claude Code session —
each in an isolated `/tmp/mm-behavioral-*` scratch project to avoid
state collisions.

Five scenarios are available:

1. **Marker visibility** — the 🧠 marker appears when a skill matches
2. **Imperative execution** — Claude invokes Bash rather than narrating
3. **Bang-command gate** — `!foo`, `/bar`, bare shell commands skip injection
4. **No-match handling** — unrelated queries emit the "no matching" marker
5. **Fresh install lifecycle** — init → seed → execute → score, end-to-end

Each scenario is self-contained in `test_behavioral.py` and can be
driven either by `pytest` or by dispatching to parallel agents. The
parallel-agent approach is faster when you want to verify all five
scenarios quickly.
