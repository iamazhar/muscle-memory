# Personal Compounding Loop Design

Date: 2026-04-23

## Purpose

`muscle-memory` should be a personal agent skill layer for individual developers.
The product promise is not "manage skills" or "run evals." The promise is:

> My coding agents remember hard-won repo-specific procedures, retrieve only the
> relevant compact playbook, and measurably reduce rediscovery and context waste.

The tool should feel like craft knowledge compounding over time. A developer who
has worked in a repo for weeks should have agents that do more with less because
the agents can draw on validated local experience instead of rediscovering the
same procedures each session.

## Product Boundary

The core product loop is:

1. Capture the developer's real task cleanly.
2. Retrieve only the best matching proven skill, if one exists.
3. Inject or emit a compact executable playbook.
4. Observe whether the task succeeded.
5. Improve, demote, or prune the skill based on evidence.
6. Show a credible proof dashboard: outcome lift, token reduction, confidence,
   and bad-skill risk.

Everything outside this loop should be hidden, removed from first-run docs, or
treated as internal machinery. Advanced concepts such as PPO, eval suites,
simulation, share/import/export, jobs, and manual review queues may continue to
exist for maintainers, but they should not define the user-facing product.

## Public Workflow

The public command surface should optimize for the personal loop:

```bash
mm init
mm use "<task>"
mm learn
mm status
mm skills
mm show <skill-id>
mm doctor
```

`mm use "<task>"` is the key missing primitive. It should retrieve the relevant
skill for a task, format it as compact executable context, and record that the
skill was offered for later attribution. For Codex, this becomes the first-class
workflow: run `mm use "fix failing pytest import path"` before prompting Codex,
then paste or pipe the generated context. For Claude Code with hooks installed,
`mm use` can preview what would activate or provide a manual fallback, but most
Claude Code users should not need to run it every turn.

`mm retrieve` is technically accurate but product-wrong. It sounds like a search
command rather than a behavior change. Keep it as a hidden compatibility alias,
but document `mm use` as the action users take.

`mm learn` should mean "turn recent sessions or transcripts into better future
behavior." It should hide bootstrap, ingest, and extract details on the happy
path. Claude Code users can run `mm learn` to scan recent local history. Codex
users can run `mm learn --transcript <file> --prompt "<task>"` until richer Codex
session discovery exists.

`mm status` should become a proof screen, not a generic stats screen. It should
answer:

```text
Is muscle-memory helping?
Outcome lift: +18% success on repeated tasks
Token reduction: 31% fewer input tokens when skills activate
Confidence: medium, 42 comparable episodes
Risks: 3 unknown outcomes, 1 stale skill
Next: run `mm learn` or inspect `mm show abc123`
```

`mm skills` and `mm show` remain trust and inspection tools. Their defaults
should highlight proven, useful, stale, and problematic skills rather than make
the user manage a raw table.

## Architecture

The biggest product gap is measurement credibility. The tool cannot claim
"better outcomes with fewer tokens" unless the data model records the actual
loop, not just inferred episodes and sidecar activation files.

Add three first-class records:

### Task

Captures the user's real intent across harnesses:

- raw prompt
- cleaned prompt
- optional intent summary
- harness
- project path
- session id
- timestamp

This fixes noisy prompt capture and gives retrieval, learning, and evaluation a
stable object to operate on.

### Activation

Records every time a skill was offered or injected:

- task id
- skill id
- retrieval distance
- final rank
- delivery mode: `claude-hook`, `codex-use`, or `manual`
- injected token count
- credited outcome when known

This should replace activation sidecar files as the canonical activation record.

### Measurement

Stores outcome and token evidence:

- outcome
- outcome confidence
- reason
- input tokens
- output tokens
- injected skill tokens
- transcript and tool-call counts
- whether the task is comparable to previous similar tasks

This lets `mm status` answer the product question directly: when a skill
activates for a repeated kind of task, does the user get a better result with
less context than similar tasks without the skill?

The core engine should stay harness-agnostic:

```text
harness adapter
  -> task capture
  -> retrieval
  -> activation delivery
  -> episode ingest
  -> outcome/token measurement
  -> skill governance
```

Claude Code and Codex should differ only at the adapter and delivery layers. The
memory layer, scoring, pruning, and status dashboard should be shared.

## Product Changes

Improve:

- Make `mm use "<task>"` the core Codex path and documented manual path.
- Make `mm status` prove value instead of listing internals.
- Capture clean task intent before retrieval and learning.
- Store activations in SQLite, not sidecar files.
- Track real token counts when transcripts provide them.
- Make skill text shorter, more imperative, and tied to repo-specific outcomes.

Remove or hide:

- Public emphasis on PPO, simulation, eval suites, sharing, import/export, jobs,
  and manual review queues.
- Estimated efficiency gain as a headline metric until actual token accounting
  exists.
- `retrieve` as the primary verb. Keep it as hidden or advanced compatibility.

Add:

- `Task`, `Activation`, and `Measurement` records.
- Confidence labels for outcome and token claims: low, medium, high.
- A first-run demo path that shows the loop in under five minutes.
- A paste-ready or copy-friendly output mode for Codex.
- A migration/backfill path for existing stores.

Refine:

- Rewrite the README around personal compounding agent skill, not research
  lineage.
- Explain skill lifecycle as trust control, not governance.
- Give Claude Code and Codex the same mental model, with different runtime
  integration depth.

## Testing

Tests should map directly to the product claim.

Core test coverage:

- Task capture tests for Claude wrapper noise, Codex JSONL, prompt override, and
  empty or noisy prompts.
- Activation attribution tests proving that a retrieved skill creates an
  activation record and later gets credited only when the matching task or
  episode succeeds.
- Token measurement tests proving that real usage fields are stored when present
  and estimates are clearly labeled when not.
- Status dashboard tests proving that output distinguishes proven value,
  insufficient evidence, unknown outcomes, stale skills, and harmful skills.
- CLI UX tests proving that `mm use`, `mm learn`, and `mm status` are the
  documented happy path for both Claude Code and Codex.

## Milestones

### Milestone 1: Measurement Credibility

- Add `tasks`, `activations`, and `measurements` tables.
- Route Claude Code hook retrieval and `mm use` through those tables.
- Add Codex-friendly `mm use "<task>"`.
- Capture token counts from Codex transcript usage records and any Claude
  transcript fields available.
- Change `mm status` to show confidence-labeled proof instead of presenting
  estimated efficiency as fact.

### Milestone 2: Skill Quality And Trust

- Tighten extractor output toward shorter executable playbooks.
- Move sidecar activation data into SQLite.
- Add backfill for existing stores.
- Improve stale and harmful skill demotion.
- Make status guidance actionable without exposing internal machinery.

### Milestone 3: Distribution And Onboarding

- Rewrite the README around the personal compounding loop.
- Add a five-minute demo that proves one repeated task improves.
- Keep advanced commands documented separately from the main path.

## Non-Goals

- Team skill sharing.
- Cloud sync.
- Multi-user governance.
- Public leaderboards or benchmark claims.
- Codex automatic prompt hooks unless a supported integration point exists.
- Treating estimated token savings as proven token savings.

## Success Criteria

The next implementation should be considered successful when:

- A new individual developer can install the tool, run `mm init`, use either
  Claude Code or Codex, and understand the loop in under five minutes.
- A Codex user has a first-class `mm use "<task>"` path that records activations
  and emits compact context.
- `mm status` can honestly say whether there is enough evidence to prove value,
  and can distinguish actual token accounting from estimates.
- The main docs no longer sell internal research machinery as the product.
- Existing stores migrate without losing episodes, skills, or activation
  history.
