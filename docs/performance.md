# Performance and cost

This document captures measured performance and cost characteristics
of `muscle-memory` as of v0.2.0.dev0, along with the optimizations
already applied and the opportunities deliberately deferred.

Measurements were taken on macOS (Apple Silicon), Python 3.12, with
`fastembed` v0.4 using the default `BAAI/bge-small-en-v1.5` model and
`sqlite-vec` v0.1.9.

## Headline numbers

| Operation | Latency | Notes |
|---|---|---|
| Pure `mm version` | ~100ms | Python interpreter + typer/rich imports |
| `mm hook user-prompt`, empty store | ~100ms | Short-circuits fastembed load (v0.2 opt) |
| `mm hook user-prompt`, seeded store | ~280ms | Dominated by fastembed first-embed (~170ms) |
| `mm hook stop` | ~110ms | Parses transcript, infers outcome, fires async children |
| KNN retrieval, 10 skills | 3.5ms median | sqlite-vec + rerank |
| KNN retrieval, 2000 skills | 3.5ms median | **flat scaling** |
| Peak RSS (hook with fastembed) | ~320MB | Dominated by ONNX runtime + BGE-small model |

| LLM operation | Wall time | Tokens (in/out) | Cost |
|---|---|---|---|
| Extraction (per session) | ~5s | 1.6k / 265 | **$0.009** |
| Refinement: gradient | ~5s | 2.0k / 256 | $0.010 |
| Refinement: rewrite | ~2s | 813 / 142 | $0.005 |
| Refinement: judge (per episode) | ~3.4s | 1.3k / 90 | $0.005 |
| **Full refinement** (3 judges) | ~16s | ~6k total | **$0.030** |

## Monthly cost projection

| Usage profile | Extractions/mo | Refinements/mo | Total/mo |
|---|---|---|---|
| Light (5 sessions/day) | 150 | ~3 | **~$1.45** |
| Heavy (20 sessions/day) | 600 | ~10 | **~$5.57** |
| Extreme (50 sessions/day) | 1500 | ~20 | **~$14.10** |

At Sonnet 4.6 pricing ($3/MTok in, $15/MTok out). Extraction cost
scales linearly with session length; longer trajectories cost more.
The elision cap at 400 tool calls keeps the worst case bounded.

## Why these numbers are fine

1. **The 280ms user-prompt hook is below the perceptual threshold.**
   Humans start to notice latency around 200-300ms; muscle-memory's
   hook sits right at the edge. It's invoked before Claude's own LLM
   call, which itself takes 3-15 seconds, so the hook is less than
   10% of total turn latency.

2. **Retrieval is flat up to at least 2000 skills.** sqlite-vec's
   exact KNN on 384-dim float vectors is ~3.5ms regardless of store
   size in this range. We never expect a project to accumulate
   thousands of skills, but if it did, retrieval wouldn't slow down.

3. **Extraction and refinement are cheap enough to be "invisible".**
   A heavy user running 20 sessions/day spends less than $6/month on
   muscle-memory's LLM calls. Refinement specifically is <$0.05 per
   skill and runs at most once per day per skill under auto-trigger.

## The dominant cost: fastembed first-embed

Breaking down the 280ms hook latency:

| Stage | Time |
|---|---|
| Python interpreter + `import muscle_memory` | ~60ms |
| + cli module (typer, rich) | ~40ms |
| + Store open + sqlite-vec extension load | ~10ms |
| + `fastembed` class construction (lazy) | ~0ms |
| + **fastembed model load + first embed call** | **~170ms** |
| + vec KNN + context format | ~3ms |

The fastembed cost is a **cold start per subprocess**: every
`UserPromptSubmit` hook invocation spawns a fresh Python process that
has to re-load the ONNX model from disk into an in-process
inference runtime. OS-level file cache keeps the `.onnx` bytes warm
across invocations (the first-ever invocation after a cold boot takes
several seconds longer), but we can't share loaded model state across
processes without a persistent daemon.

## Optimizations applied in v0.2+

### Empty-store short-circuit (retriever.py)

**Before:** every hook invocation loaded fastembed, even with zero
skills in the store. The first few sessions after `mm init` paid a
170ms cost for no benefit.

**After:** `Retriever.retrieve()` checks `store.count_skills() == 0`
before constructing any embedder query. Empty-store hooks now return
in ~100ms, identical to `mm version`.

**Impact:** ~180ms savings on every `UserPromptSubmit` until the store
starts accumulating skills. Larger impact on fresh projects or short
bootstraps.

### Lexical prefilter for obvious no-match prompts

**Before:** unrelated natural-language prompts still paid the embedding
cost before being filtered out downstream.

**After:** the retriever performs a cheap token-overlap precheck against
trusted skill activations. When a prompt has no lexical overlap with any
trusted skill, retrieval skips embedding entirely and records that decision
in `MM_DEBUG` telemetry.

**Impact:** obvious no-match prompts now avoid the cold embed path and
become visible in operator tooling (`mm doctor`, `mm stats`) as lexical
prefilter skips.

### Shell-escape gate (v0.1, hooks/user_prompt.py)

Already in v0.1, but worth counting: bang commands (`!ls`), slash
commands (`/model`), and bare shell commands (`mm list`, `git status`)
skip retrieval entirely. They return in ~100ms regardless of store
size.

## Deferred optimizations

These would provide meaningful wins but weren't quick enough for v0.2.

### Persistent embedder daemon

**The big one.** Run a long-lived Python process that keeps fastembed
loaded and serves embed requests over a Unix socket. Would reduce the
hook latency from 280ms to ~20ms (cold embed time only).

**Cost of building:** a few days. Requires daemon lifecycle
management, socket protocol, fallback for when daemon isn't running,
macOS launchd / systemd integration. Not a quick win.

**When to do it:** if real dogfooding shows 280ms is noticeable, or
if a user complains, or if we ever add a second embedder user (e.g.,
LLM-judge rerank).

### Smaller embedder model

Switch from `BAAI/bge-small-en-v1.5` (33M params, ~130MB unquantized)
to something like `sentence-transformers/all-MiniLM-L6-v2` (22M params)
or `paraphrase-MiniLM-L3-v2` (17M params, 3 layers).

**Estimated savings:** maybe 50-100ms on the first-embed cost. Quality
impact: probably minor for short activation matching, but needs
measurement.

**When to do it:** together with a persistent daemon. In isolation,
saving 50ms without solving the cold-start problem isn't worth the
quality risk.

### Keyword pre-filter

Maintain an inverted index (SQLite FTS5 or an explicit words table)
mapping tokens → skill ids. At query time, tokenize the prompt, do a
cheap FTS/SQL lookup, and only fall back to embedding if there's at
least one token-level hit.

**Estimated savings:** 170ms on unrelated queries (which, for a project
with narrow skill coverage, is most queries). For queries that do
have token overlap, the FTS lookup is ~1ms and embedding still runs.

**Cost of building:** half a day. Requires schema migration for the
FTS index, rebuild during `mm init` and on insert, fallback if FTS
returns nothing.

**When to do it:** when skill stores grow past ~50 skills and most
user prompts don't match anything. Until then, fastembed is doing
useful work on every match.

### Trajectory elision for extraction

Very long sessions (our own build session hit 490 tool calls) can
cost $0.10-$0.25 per extraction versus the ~$0.01 typical case. We
already elide above 400 tool calls but keep 80 head + 120 tail; we
could be more aggressive if cost becomes an issue.

**When to do it:** when a user reports high monthly cost with many
long sessions, or when we see specific extraction calls exceed
$0.10.

### Streaming judge calls in refinement

PPO-Gate currently runs one judge call per trajectory sequentially.
For a skill with 10 trajectories, that's 10 × ~3.4s = ~34 seconds
end-to-end wall time. Running them in parallel via async would cut
this to ~3.5s.

**When to do it:** if refinement latency starts blocking anything.
Currently refinement runs async in a detached subprocess so wall
time doesn't matter to the user — only total cost matters, and
that's unchanged by parallelization.

## Re-measuring

The benchmark scripts that produced these numbers are not checked in
because they depend on live network calls to Anthropic's API. To
re-run them, see the `Bash` commands in the git history:

```bash
git log --all --format='%H %s' | grep -i "perf\|benchmark"
```

For quick ad-hoc latency checks:

```bash
# Pure startup baseline
/usr/bin/time -p mm version

# Empty-store hook (should be ~100ms after v0.2 opt)
echo '{"session_id":"x","cwd":"'"$PWD"'","prompt":"test"}' \
  | /usr/bin/time -p mm hook user-prompt >/dev/null

# Stop hook (should be ~110ms)
echo '{"session_id":"x","cwd":"'"$PWD"'","transcript_path":"/path/to/session.jsonl"}' \
  | /usr/bin/time -p mm hook stop >/dev/null
```

## Summary

Current performance is **acceptable for v0.2 dogfooding** and cost is
**invisibly cheap** ($5-6/month for heavy users). The one wart is the
280ms seeded-store hook latency, which is at the edge of perceptible
but not actually problematic in practice because it runs before
Claude's own multi-second LLM call.

If real-world use shows 280ms is noticeable, the highest-impact
optimization is the persistent embedder daemon — that's the only one
that would get us to sub-100ms hook latency across the board.
