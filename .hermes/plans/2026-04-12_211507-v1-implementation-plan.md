# muscle-memory v1 Implementation Plan

> For Hermes: use subagent-driven-development for larger follow-on tasks, but start with the smallest high-leverage reliability work first.

Goal: ship muscle-memory v1 as a dependable, fast, selective, explainable, local-first product for daily Claude Code use.

Architecture:
- Keep the current core architecture: Claude Code hooks + local SQLite store + async extraction/refinement.
- Focus v1 work on reliability, observability, hot-path performance, trust governance, and operator UX rather than expanding product scope.
- Land changes in milestone order so each phase produces a noticeably more shippable system.

Tech Stack: Python 3.11+, Typer CLI, SQLite, Claude Code hooks, fastembed, pytest, rich.

---

## v1 release milestones

1. Reliability and observability
2. Prompt-path performance
3. Trustworthy retrieval and lifecycle governance
4. Eval-driven quality governance
5. Operator UX polish
6. Packaging / install confidence

---

## Phase 1: Reliability and observability

Success metric:
- A user can answer why a skill did or did not fire.
- A user can inspect extraction/refinement failures.
- Async work is never silently lost.

### Task 1.1: Add debug logging primitives

Objective: create a single logging utility for hooks and background jobs, gated by `MM_DEBUG`.

Files:
- Create: `src/muscle_memory/debug.py`
- Modify: `src/muscle_memory/config.py`
- Test: `tests/test_debug.py`

Implementation notes:
- Add `Config.debug_enabled` derived from `MM_DEBUG`.
- Add helpers to append newline-delimited structured logs under `.claude/mm.debug.log` or equivalent DB-adjacent path.
- Log fields: timestamp, component, event, session_id, cwd, details.
- Logging failures must never crash hooks.

Validation:
- `pytest tests/test_debug.py -v`
- Verify log writes are gated on `MM_DEBUG=1`.

### Task 1.2: Instrument `UserPromptSubmit` hook

Objective: make retrieval decisions inspectable.

Files:
- Modify: `src/muscle_memory/hooks/user_prompt.py`
- Test: `tests/test_integration.py`
- Test: `tests/test_edge_cases.py`

Implementation notes:
- Log reasons for early return: malformed payload, shell escape, missing DB, paused, no hits.
- Log retrieval summary: prompt excerpt, number of hits, skill ids, distances, filtered counts if available.
- Log activation sidecar write success/failure.
- Preserve current fail-safe behavior.

Validation:
- Targeted pytest for hook integration tests.
- Manual smoke with `MM_DEBUG=1` should produce readable events.

### Task 1.3: Instrument `Stop` hook

Objective: make episode parsing, scoring, extraction spawn, and refinement spawn inspectable.

Files:
- Modify: `src/muscle_memory/hooks/stop.py`
- Test: `tests/test_integration.py`
- Test: `tests/test_edge_cases.py`

Implementation notes:
- Log transcript parse summary, inferred outcome, activated skills, stored episode id.
- Log extraction/refinement subprocess launch attempts and failures.
- Log fallback queue writes.

Validation:
- Hook tests remain green.
- Debug log shows end-to-end stop-hook lifecycle.

### Task 1.4: Add tracked async jobs

Objective: replace opaque fire-and-forget behavior with a visible job model.

Files:
- Modify: `src/muscle_memory/db.py`
- Modify: `src/muscle_memory/hooks/stop.py`
- Modify: `src/muscle_memory/cli.py`
- Create: `tests/test_jobs.py`

Implementation notes:
- Add `jobs` table with: id, kind, payload, status, attempts, created_at, updated_at, error.
- Record extraction/refinement jobs before subprocess launch.
- Mark launch failure in DB instead of relying only on queue files.
- Add CLI read surface, e.g. `mm jobs list`.

Validation:
- `pytest tests/test_jobs.py -v`
- existing tests pass

### Task 1.5: Add replay/retry workflow

Objective: make failed async work recoverable.

Files:
- Modify: `src/muscle_memory/cli.py`
- Modify: `src/muscle_memory/hooks/stop.py`
- Create: `tests/test_cli_jobs.py`

Implementation notes:
- Add `mm jobs retry <id>` and `mm jobs retry-failed`.
- Either consume `mm.extract_queue.txt` into jobs or delete the queue-file path after DB-backed retries exist.

Validation:
- CLI tests for retry flow.

### Task 1.6: Add `mm doctor`

Objective: give operators one command to inspect system health.

Files:
- Modify: `src/muscle_memory/cli.py`
- Test: `tests/test_cli_doctor.py`

Implementation notes:
- Check: db exists, paused flag, debug enabled, pending/failed jobs, hook config presence, last debug events if available.
- Output human-readable and JSON.

Validation:
- CLI tests for empty and populated projects.

---

## Phase 2: Prompt-path performance

Success metric:
- Retrieval overhead feels negligible in daily use.
- Performance is observable and stable.

### Task 2.1: Add retrieval telemetry
- Files: `src/muscle_memory/retriever.py`, `src/muscle_memory/hooks/user_prompt.py`, tests as needed
- Log: embed time, search time, rerank time, total hook time

### Task 2.2: Add lexical prefilter
- Files: `src/muscle_memory/retriever.py`, `src/muscle_memory/db.py`, retriever tests
- Use token overlap / FTS5 to skip obvious no-match embedding work

### Task 2.3: Add persistent embedder daemon (if Phase 2.2 alone is insufficient)
- Files likely: new daemon module, retriever integration, docs/performance.md
- Add graceful fallback when daemon is unavailable

---

## Phase 3: Trustworthy retrieval and lifecycle governance

Success metric:
- Low false-positive rate.
- Weak live skills can be demoted or quarantined.

### Task 3.1: Refine promotion policy
- Use more than source-episode count
- Consider runtime success quality and review state

### Task 3.2: Add demotion path
- Allow `live -> candidate` or `live -> review-needed`
- Trigger on repeated poor runtime behavior

### Task 3.3: Improve review UX
- Batch review
- explain why candidate/live/proven
- show evidence summary

---

## Phase 4: Eval-driven quality governance

Success metric:
- evaluator outputs influence promotion, refinement, and pruning.

### Task 4.1: Feed relevance/adherence/correctness into scoring
### Task 4.2: Gate `proven` on measured quality, not raw success ratio alone
### Task 4.3: Use eval signals to prioritize refinement and demotion

---

## Phase 5: Operator UX polish

Success metric:
- New users can inspect and operate the system entirely from CLI.

### Task 5.1: Improve `mm stats`
### Task 5.2: Add explanation surfaces for skill firing / suppression
### Task 5.3: Polish review, jobs, and doctor commands

---

## Phase 6: Packaging and release confidence

Success metric:
- install, init, bootstrap, and dogfood work consistently on supported platforms.

### Task 6.1: tighten install/bootstrap docs
### Task 6.2: ensure macOS/Linux happy path coverage
### Task 6.3: update release checklist and v1 release notes

---

## Immediate execution plan

Start with Phase 1, Task 1.1 and 1.2.

Why:
- highest leverage
- smallest implementation slice
- unlocks everything else by making current behavior visible

### First execution slice
1. Add `src/muscle_memory/debug.py`
2. Add config support for `MM_DEBUG`
3. Add tests for debug utility
4. Instrument `hooks/user_prompt.py`
5. Run focused tests
6. Then continue to stop-hook instrumentation

---

## Files likely to change across v1

Core runtime:
- `src/muscle_memory/config.py`
- `src/muscle_memory/cli.py`
- `src/muscle_memory/db.py`
- `src/muscle_memory/retriever.py`
- `src/muscle_memory/hooks/user_prompt.py`
- `src/muscle_memory/hooks/stop.py`

Learning / governance:
- `src/muscle_memory/scorer.py`
- `src/muscle_memory/admission.py`
- `src/muscle_memory/refine.py`
- `src/muscle_memory/eval/evaluator.py`
- `src/muscle_memory/eval/scorers.py`

New likely modules:
- `src/muscle_memory/debug.py`
- `src/muscle_memory/jobs.py` or DB-backed job helpers

Tests:
- `tests/test_integration.py`
- `tests/test_edge_cases.py`
- `tests/test_retriever.py`
- `tests/test_cli_stats.py`
- `tests/test_refine.py`
- `tests/test_debug.py`
- `tests/test_jobs.py`
- `tests/test_cli_doctor.py`
- `tests/test_cli_jobs.py`

Docs:
- `README.md`
- `docs/testing.md`
- `docs/performance.md`
- `CHANGELOG.md`

---

## Risks / tradeoffs

- Hook logging must remain fail-safe and low-overhead.
- Background-job tracking adds schema and lifecycle complexity; keep it minimal.
- Lexical prefilters can reduce recall if too strict.
- Eval-governed trust changes may alter current dogfood behavior and will need careful rollout.

---

## Verification strategy

Per phase:
- keep existing pytest suite green
- add focused tests for each new CLI/runtime behavior
- update docs only after code matches reality

Release gate for v1:
- reliability milestone complete
- performance milestone complete
- trust/lifecycle governance complete
- operator surfaces (`stats`, `doctor`, jobs, review) are usable
- docs match shipped behavior
