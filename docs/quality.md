# Skill Quality Gates

`muscle-memory` should be conservative about what enters the retrievable skill pool.

## Product policy

False negatives are cheaper than false positives.

Missing one potentially useful skill is acceptable. Injecting junk skills teaches
users to distrust the system.

## Current admission gates

Extracted skills are only admitted when all of the following are true:

1. The source episode is a clear `success`.
2. The source episode is non-trivial and contains at least 2 tool calls.
3. The extracted activation is specific enough to describe a reusable trigger.
4. The execution contains at least 2 concrete steps.
5. The skill text does not contain obvious one-off literals such as:
   - temp paths
   - session IDs / UUIDs
   - PR / issue numbers
   - exact dates

If any of those checks fail, the candidate is dropped before insertion.

## Retrieval lifecycle

Passing admission does not make a skill immediately retrievable.

- `candidate` skills are quarantined by default.
- Candidates can be promoted to `live` after repeated evidence from distinct
  successful source episodes, or by explicit human approval.
- Only `live` and `proven` skills are eligible for automatic retrieval.

## Why this is strict

The system already has later defenses like retrieval filters, scoring, refinement,
and pruning. But the cheapest place to stop junk is at admission time, before a
candidate can ever be trusted.
