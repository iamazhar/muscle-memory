# Changelog

All notable changes to `muscle-memory` will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.2.0.dev

### Added — Non-Parametric PPO refinement loop

The headline feature of ProcMEM (arxiv:2602.01869) that v0.1 deferred.
Three-stage skill refinement that fixes broken skills in place instead
of waiting for accidental re-extraction from new trajectories.

- **`muscle_memory/refine.py`** — new module implementing:
  - `extract_gradient(skill, episodes, llm) → SemanticGradient` —
    Stage 1 hindsight attribution: one LLM call analyzes a skill plus
    its recent trajectories and produces per-field feedback describing
    what's wrong and why.
  - `apply_gradient(skill, gradient, llm) → Skill` — Stage 2 rewrite:
    a second LLM call edits the skill text in place, preserving
    unchanged fields verbatim and bumping `refinement_count`.
    Saves the previous text to `previous_text` for rollback.
  - `verify_refinement(original, revised, episodes, llm) → bool` —
    Stage 3 PPO Gate (LLM-judge proxy): for each stored trajectory,
    ask the LLM to score the revision against the original on a
    -2..+2 scale; accept only if mean ≥ 0.5 AND no single verdict
    strongly regresses (-2 is a veto). Adapts the paper's
    importance-ratio + clipped-surrogate formula to an API that
    doesn't expose token logprobs.
  - `refine_skill(skill, episodes, llm, store=)` — orchestrator that
    runs all three stages; never raises (returns `RefinementResult`
    with `accepted=False` + `rejection_reason` on any failure).
  - `should_auto_refine(skill)` — trigger criteria: ≥5 invocations,
    ≥2 failures, score ≤ 0.6.

- **Three new version-controlled prompts** in
  `src/muscle_memory/prompts/`:
  - `refine_gradient.md` — hindsight attribution prompt with explicit
    discipline against vague feedback and a `should_refine` gate.
  - `refine_rewrite.md` — surgical-edit prompt that preserves
    unchanged fields and keeps imperative framing.
  - `refine_judge.md` — counterfactual judge with a strict -2..+2
    rubric anchored to specific trajectory evidence.

- **`mm refine` CLI command** with three modes:
  - `mm refine <skill-id>` — refine one specific skill
  - `mm refine --auto` — sweep all skills meeting auto-refine criteria
  - `mm refine <id> --dry-run` — preview changes without persisting
  - `mm refine <id> --rollback` — restore the previous skill text
    from the stored `previous_text`

- **Async refinement triggered from the Stop hook** — after each
  episode, if any skill now meets `should_auto_refine`, the Stop
  hook fires a detached `mm refine --auto` subprocess alongside the
  existing `mm extract-episode` call. Both are fire-and-forget so
  the hook returns immediately.

- **Schema migration to v2** — `skills` table gains two new columns:
  `refinement_count INTEGER NOT NULL DEFAULT 0` and
  `previous_text TEXT` (JSON-encoded). Handled via idempotent
  `ALTER TABLE ADD COLUMN` migration in `_migrate()`.

- **`Store.find_episodes_for_skill(id, limit)`** — scan recent
  episodes for ones where the given skill was activated. Used by
  the refinement loop to gather evidence.

### Validated — end-to-end dogfood

Ran the full refinement loop with real Sonnet 4.6 calls against a
seeded skill that had a known bug (`python` instead of `python3` in
the verification step). Four stored episodes where every one showed
the same `command not found: python` recovery pattern. Result:

```
Refining 3dfd5789 (2/6 successes, score 0.33)
  ✓ accepted (mean judge +2.00 across 4 trajectories)
    root cause: Step 3 calls `python` which is not available on macOS
    PATH, causing a `command not found` error in every observed episode
    before Claude retried with `python3`.
```

The skill was revised in place; only step 3 changed from `python -c`
to `python3 -c`; activation and termination untouched;
`refinement_count` bumped to 1; `previous_text` preserved for
rollback. This is the paper's core contribution working on a real
coding-agent bug.

### Added — tests

- **22 new tests** in `tests/test_refine.py`:
  - `TestExtractGradient` (5) — happy path, no-change response,
    LLM failure propagation, zero-episodes guard, malformed response
  - `TestApplyGradient` (3) — preserves unchanged fields + bumps
    count, LLM failure propagation, empty-response fallback
  - `TestVerifyRefinement` (6) — accept on positive mean, strong-
    regression veto, low-mean rejection, no-textual-changes early
    exit, no-trajectories guard, mid-judge error handled as neutral
  - `TestRefineSkillOrchestrator` (4) — full accept path with
    store persistence, `should_refine=false` early return, PPO Gate
    rejection does NOT persist, LLM crash does not raise
  - `TestAutoRefineTrigger` (4) — fires for low-scoring skill with
    data, silent on high-success skills, silent without invocations,
    silent without failures

### Changed

- `pyproject.toml` version bumped to `0.2.0.dev0`.
- Default model for extraction + refinement is Sonnet 4.6 (unchanged
  from v0.1). Refinement uses the same LLM provider as extraction;
  no separate configuration.

---

## [0.1.0] — 2026-04-07

First public release. Implements the core thesis from
[ProcMEM (arxiv:2602.01869)](https://arxiv.org/abs/2602.01869) — executable
procedural memory for LLM agents — specialized for Claude Code and ready
to install with `uv tool install muscle-memory`.

### Added — core pipeline

- **Skill model** as the paper's ⟨activation, execution, termination⟩
  natural-language tuple, with practical metadata fields added on top:
  scoring, invocations, successes, failures, maturity, source episodes,
  tags, tool hints, and project/global scope.
- **SQLite + sqlite-vec storage** with KNN search over activation
  embeddings and WAL mode for concurrent readers/writers.
- **Pluggable embedder**: default `fastembed` (BGE-small-en-v1.5, local
  CPU, 384 dims, no API key needed) with OpenAI and Voyage as optional
  extras.
- **Pluggable LLM**: default Anthropic Claude Sonnet 4.6 for extraction
  work, with OpenAI `gpt-4o-mini` available via `uv tool install
  'muscle-memory[openai]'` + `MM_LLM_PROVIDER=openai`.
- **Conservative LLM-driven extractor** with a version-controlled prompt
  that produces zero-to-few skills per episode; imperative output format
  (`"Run X"`, not `"You should check for X"`); explicit anti-copy
  instruction to prevent anchoring on prompt examples.
- **Heuristic outcome inference**: final-state-dominates logic over
  trajectory tool calls. Handles mid-session failures with end-of-session
  recovery as SUCCESS (not FAILURE), recognizes Python `__init__.py`
  paths as success signals, treats `ModuleNotFoundError` /
  `command not found` / `Permission denied` as failure signals.
  Skill-activated sessions bias toward SUCCESS when nothing obviously
  failed — without this bias, pure-Bash success chains land in UNKNOWN.
- **Fast retriever** (<500ms target) using local embedding + sqlite-vec
  KNN, with maturity and score bonuses in the reranking.
- **Skill-activated visibility marker**: the injection wrapper
  instructs Claude to prefix responses with
  `🧠 **muscle-memory**: executing playbook — <title>` so users can see
  the system working in real time.
- **Imperative framing**: the injection wrapper explicitly tells Claude
  to EXECUTE playbook steps (run the commands, make the edits) rather
  than narrate them. This was crucial — without it, Claude treated
  skills as advice and echoed them back to the user instead of acting.
- **Credit assignment + Darwinian pruning**: `Scorer.credit_episode`
  updates activated-skill counts from episode outcome; pool maintenance
  prunes skills with score ≤ 0.2 after ≥ 5 invocations and enforces a
  configurable capacity (default 500).
- **Embedding-based deduplication** at two levels: `add_skill_with_dedup`
  gates every insertion (prevents async-extract bloat), and
  `find_near_duplicate_groups` + `consolidate_group` merge existing
  near-duplicates into the best-scoring member while preserving all
  historical counts via sum/union.

### Added — integration and UX

- **Claude Code hook integration** via `.claude/settings.json`:
  `UserPromptSubmit` → retrieve + inject top playbooks,
  `Stop` → parse transcript, infer outcome, credit skills, fire async
  extraction in a detached subprocess.
- **`mm init`** command wires hooks non-destructively — preserves any
  existing hooks already configured in `settings.json`, is idempotent
  (no duplicates on re-run), scope-aware (project or global).
- **`mm bootstrap`** seeds the store from existing Claude Code session
  JSONL files under `~/.claude/projects/<encoded-path>/`, optionally
  filtered by project and time window.
- **Shell-escape gate**: `UserPromptSubmit` hook detects bang commands
  (`!ls`), slash commands (`/model`), and bare shell commands (`mm list`,
  `git status`), and short-circuits retrieval for them. Prevents
  invocation-count bloat from scripting-style prompts.
- **Hook failure is always silent**: both hooks wrap the entire body in
  `try/except Exception` + `isinstance(dict)` guards so a hook crash
  can never break a user's Claude Code session. Tested against 9+
  garbage-input cases.

### Added — CLI

- `mm init`, `mm list`, `mm show`, `mm stats`, `mm export`, `mm import`
- `mm bootstrap` — seed from existing Claude Code session history
- `mm prune` — remove demonstrably bad skills below a score floor
- `mm dedup` — collapse near-duplicate clusters (with `--dry-run` and
  `--threshold`)
- `mm rescore` — re-run the outcome heuristic on every stored episode
  and re-credit skills; useful after upgrading
- `mm hook user-prompt` / `mm hook stop` — hook handlers (not for
  direct use, invoked by Claude Code)
- `mm extract-episode <id>` — hidden command used by the async
  extraction pipeline

### Added — tests and docs

- **99 pytest tests** covering models, storage, outcomes, extraction,
  retrieval, dedup, integration, and edge cases (malformed input,
  unicode, SQL-injection-shaped strings, concurrency, 500-skill KNN
  latency).
- **5 opt-in behavioral tests** (`test_behavioral.py`) driving real
  `claude -p` sessions in scratch projects, covering visibility marker,
  imperative execution, shell-escape gate, no-match handling, and full
  install lifecycle.
- **`docs/authentication.md`** — auth options, per-provider env vars,
  the Max/Pro subscription limitation.
- **`docs/testing.md`** — test layers, the `claude -p` final-text-only
  gotcha and three workarounds, layered-safety model.
- **`docs/development.md`** — dev workflow, including the macOS + uv +
  Python 3.12 hidden-`.pth` gotcha and `scripts/dev-sync` fix.

### Known limitations

- **No Non-Parametric PPO refinement loop.** The paper's killer
  feature — `ω' = ω ⊕ ḡ_ω` rewriting skills from semantic gradients
  with PPO-Gate trust-region verification — is not implemented in
  v0.1. Skill "refinement" happens accidentally via re-extraction
  from new trajectories, which is lossy and late. This is the primary
  v0.2 target.
- **No Q-value ranking**. The retriever ranks by embedding distance +
  maturity/score bonus, not by estimated Q-value per the paper's
  "Selection by Value" alternative.
- **No cross-model transfer tests**. The paper demonstrates Gemma-2-9B
  → Qwen3 → LLaMA transfer. We pin to one provider per install.
- **Max/Pro subscription auth is not reusable**. `muscle-memory` needs
  a real Anthropic API key (with billing credits) or an OpenAI key for
  extraction. Claude Code subscription auth is OAuth-based and not
  exposed to third-party tools.
- **macOS + uv + Python 3.12 hidden `.pth`**. Development workflow
  requires `chflags nohidden .venv/lib/python*/site-packages/*.pth`
  after every `uv sync`. Handled by `scripts/dev-sync`. Does not
  affect end users who install via `uv tool install`.
- **Marker visibility in `claude -p`**. The `🧠` marker is emitted
  correctly but `claude -p` returns only the final assistant text,
  so multi-turn executions drop the marker from stdout. Interactive
  Claude Code users see it every time. Documented in `docs/testing.md`.

[0.1.0]: https://github.com/iamazhar/muscle-memory/releases/tag/v0.1.0
