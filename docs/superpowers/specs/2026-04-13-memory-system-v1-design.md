# Memory System V1 Release Design

Date: 2026-04-13
Project: `muscle-memory`
Scope: release design for a Claude Code-first `v1`

## Goal

Make `muscle-memory` safe and useful enough for daily Claude Code use that it can be released as a `v1`.

For this release, the primary objective is not maximum autonomy. It is obvious day-to-day user value inside a boundary that stays trustworthy under normal failures, wrong matches, and incomplete learning.

## Product Position

`v1` is a Claude Code-first memory system that:

- automatically retrieves trusted skills during normal prompts
- automatically extracts new candidate skills from successful sessions
- improves over time without forcing the user to manage memory manually
- remains inspectable and reversible when memory quality is uncertain

`v1` is not the promise that every harness is equally supported, that every repeated task will be learned immediately, or that autonomous refinement can run without guardrails.

## Product Boundary

The supported surface for `v1` is Claude Code only.

What is in scope:

- Claude Code hook installation and runtime integration
- automatic retrieval on prompt submission
- automatic extraction from successful Claude Code sessions
- candidate quarantine, conservative promotion, and governance
- bounded automatic refinement with verification and rollback
- user-facing control surfaces such as `review`, `doctor`, `stats`, `rollback`, and pause/resume behavior
- release documentation, release preflight, and a demo path that makes first success likely

What is out of scope for the `v1` promise:

- generic or multi-harness support as a first-class release story
- aggressive autonomous promotion of untrusted skills
- broad claims about perfect retrieval across all repositories and task shapes
- enterprise deployment, sync, or multi-user collaboration features

## Recommended Approach

The recommended release posture is guarded autopilot.

This means:

- automatic learning is enabled so the product feels useful without babysitting
- trust transitions are conservative so fresh learning cannot immediately pollute retrieval
- retrieval-time trust is separated from learning-time trust
- failures degrade to normal Claude Code behavior rather than a broken session

Alternative approaches were considered and rejected for `v1`:

- full autonomy first: strongest demo, weakest trust boundary
- manual trust pipeline: safest, but too much friction for the desired daily-use value

## Trust Model

The runtime loop for `v1` is:

1. The Claude Code prompt hook receives the current prompt.
2. The retriever returns only `live` and `proven` skills.
3. Claude receives a small injected playbook set and decides whether any playbook clearly applies.
4. Matching playbooks are executed in the normal Claude Code session.
5. The stop hook records outcomes and activated skills.
6. Extraction, scoring, governance, and refinement run in the background and affect only future retrieval through explicit trust transitions.

This split is the core `v1` safety property. Newly extracted skills may be learned automatically, but they begin as `candidate` and are never auto-injected until they have earned trust.

### Trust guarantees

`v1` should guarantee all of the following:

- Hook failures do not break Claude Code sessions.
- Missing databases, bad payloads, paused state, or background-job failures degrade to "no memory help" rather than user-visible failure.
- Candidates are never automatically retrieved.
- Promotions require repeated successful evidence from distinct episodes or explicit human approval.
- Weak trusted skills can be flagged for review or demoted by governance.
- Refinement is reversible through rollback, limited to per-skill rewrites, and does not silently skip verification.

### Automation policy

The default `v1` automation policy should be:

- automatic retrieval: on, for trusted skills only
- automatic extraction: on
- automatic promotion from candidate: on, but conservative
- automatic governance: on
- automatic refinement: on, but limited to verified per-skill rewrites with rollback
- automatic retrieval of candidates: off, always

## V1 Release Gate

The product is `v1 ready` only if it passes all four gates below.

### 1. User trust

A new Claude Code user must be able to:

- install the tool
- initialize it in a project
- see a repeated workflow become easier
- understand what happened without reading source code
- recover when memory is wrong or noisy

This requires:

- visible execution markers when a playbook fires
- a clear review surface for candidates and weak skills
- obvious controls for `pause`, `doctor`, `review`, `rollback`, and `stats`
- docs that explain the trust lifecycle and failure recovery in plain language

### 2. Runtime reliability

The Claude Code integration must be safe to leave on during normal work.

This requires:

- silent hook failure semantics
- prompt-hook latency that stays within the documented budget
- safe behavior when retrieval is wrong, including wrapper guidance and Claude judgment declining bad playbooks
- background jobs that can fail without corrupting trust state or breaking foreground use
- confidence in the full Claude Code lifecycle: install, prompt hook, stop hook, extraction, scoring, refinement, governance

### 3. Memory quality

The retrievable pool must remain conservative enough that users mostly see relevant memory instead of noise.

This requires:

- strict admission filters
- quarantine-by-default for new skills
- conservative promotion thresholds
- duplicate control and pruning
- governance for weak trusted skills
- release evaluation against a benchmark that measures:
  - retrieval precision on repeated real tasks
  - false-positive rate on unrelated prompts
  - execution success rate for matched skills
  - candidate-to-live promotion quality

The implementation plan must define explicit pass thresholds for those benchmark metrics so release readiness is not decided ad hoc.

### 4. Operational readiness

Releasing `v1` must not depend on tribal knowledge.

This requires:

- aligned version metadata and changelog entries
- reproducible distribution artifacts
- passing release preflight checks
- current release notes and install docs
- a usable demo or dogfood path
- clear documentation of what is supported versus experimental
- a short release checklist for future cuts

## Concrete Ship Criteria

The project should not ship `v1` until these are true:

- Claude Code install and dogfood flow works end-to-end on a clean Claude Code setup.
- The fast automated test suite passes.
- The Claude Code behavioral tests that defend the release story pass.
- Release workflow and artifact verification tests pass.
- Retrieval and hook performance stay inside the documented budget.
- Documentation covers install, first-use, trust lifecycle, recovery, and known limitations.
- Generic harness support is clearly documented as secondary or post-`v1`.
- Release preflight produces valid artifacts and checksums without manual intervention.

## Non-Goals

The following are explicitly not required to declare `v1`:

- broad support for every agent harness
- perfect semantic retrieval across unusual prompts
- zero manual review forever
- large-scale observability, sync, or collaboration features

## Design Implications For Implementation Planning

The implementation plan should focus on closing the highest-risk gaps against the release gate rather than adding new product surface area.

Priority order:

1. Defend Claude Code runtime reliability and failure semantics.
2. Tighten trust and quality controls around retrieval, promotion, and refinement.
3. Make first-run and recovery UX obvious in docs and CLI flows.
4. Prove the release path with preflight, tests, and launch checklist coverage.

## Open Decisions Resolved In This Design

The following decisions are fixed for `v1` and should not be reopened during planning unless new evidence appears:

- success criterion: production-ready for daily use
- product priority: obvious user value over maximal inspectability or maximal conservatism
- supported surface: Claude Code only
- automation default: automatic, with conservative trust boundaries

## Summary

`muscle-memory v1` should ship as a Claude Code-first, guarded-autopilot memory system. It should feel automatic and valuable in normal use, but only because trust transitions, runtime failure semantics, and recovery workflows are strong enough that mistakes degrade safely. The release is ready when Claude Code users can install it, feel the value quickly, understand what it is doing, and recover cleanly when memory quality is imperfect.
