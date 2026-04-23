# Authentication

`muscle-memory` needs an LLM for two tasks:

1. **Skill extraction** — runs after a session ends, reads the
   trajectory, decides whether any reusable Skills should be extracted.
   One call per session. Cheap (typically <$0.001 with Haiku 4.5).
2. **Retrieval rerank** — *not implemented in v1*, but reserved for
   future LLM-judge reranking of top-k skill matches.

Neither the runtime retriever (embedding search) nor the SQLite layer
needs any LLM — those are fully local.

## Known limitation: Claude Code subscription auth is not reusable

If you use **Claude Code** via a Max or Pro subscription, that auth is
OAuth-based and scoped to Claude Code itself. There is currently no
supported way for a third-party tool to reuse that session to call the
Anthropic API. We will not ship hacks to steal OAuth tokens out of
`~/.claude/`; they are fragile and violate the auth contract.

If Anthropic ever releases an official SDK that can authenticate as a
Claude Code subscription, we will adopt it immediately.

**In the meantime**, you need either:

- A separate Anthropic API key with billing credits, **or**
- An OpenAI API key, **or**
- A local LLM (Ollama support is planned for v2).

## Option 1 — Anthropic API key (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
mm learn
```

You can top up at <https://console.anthropic.com/settings/billing>.
$5 goes a long way; a typical session extraction with Haiku 4.5 costs
around a tenth of a cent.

## Option 2 — OpenAI

```bash
# install with the openai extra
uv tool install 'muscle-memory[openai]'

export OPENAI_API_KEY=sk-...
export MM_LLM_PROVIDER=openai
# optional — defaults to gpt-4o-mini
export MM_LLM_MODEL=gpt-4o-mini

mm learn
```

## Option 3 — Ollama (planned)

Not yet supported. Track progress in the v2 milestone.

## Advanced: different keys per project

Set `MM_LLM_API_KEY` per shell / per project:

```bash
# project A uses your personal key
cd ~/projects/a
export MM_LLM_API_KEY=$PERSONAL_ANTHROPIC_KEY

# project B uses the work team key  
cd ~/projects/b
export MM_LLM_API_KEY=$WORK_ANTHROPIC_KEY
```

`MM_LLM_API_KEY` wins over the provider-specific env var
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) if both are set.
