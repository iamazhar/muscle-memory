"""Fast skill retrieval for the user-prompt hook.

Must stay under a ~500ms budget — this runs synchronously before
every user turn. We do:

  1. Embed the query (fastembed locally, ~20-100ms)
  2. KNN lookup in sqlite-vec (<5ms)
  3. Distance floor + maturity/score reranking (no LLM)

The result is a small, ordered list of Skills that the hook
formats into an `<additional_context>` block.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from muscle_memory.config import Config
from muscle_memory.db import Store
from muscle_memory.embeddings import Embedder
from muscle_memory.models import Maturity, Skill

# maturity weight — proven skills beat candidates at equal distance
_MATURITY_BONUS: dict[Maturity, float] = {
    Maturity.CANDIDATE: 0.00,
    Maturity.LIVE: 0.05,
    Maturity.PROVEN: 0.10,
}

_STRONG_MATCH_MAX_RANK = 0.75
_WEAK_MATCH_MAX_RANK = 1.0
_SHORT_TECH_TOKENS = {"ai", "ci", "db", "go", "js", "mm", "os", "py", "ui", "uv"}
_STOPWORDS = {
    "a",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "can",
    "code",
    "current",
    "do",
    "does",
    "for",
    "from",
    "get",
    "help",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "please",
    "project",
    "repo",
    "session",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "to",
    "turn",
    "user",
    "using",
    "wants",
    "when",
    "where",
    "which",
    "with",
    "your",
}


@dataclass
class RetrievedSkill:
    skill: Skill
    distance: float
    score_bonus: float

    @property
    def final_rank(self) -> float:
        """Lower is better. Distance minus bonuses."""
        return self.distance - self.score_bonus


def _normalize_token(token: str) -> str:
    token = token.lower().strip("`'\"")
    if len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 5 and (token.endswith("ed") or token.endswith("es")):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_]+", text):
        token = _normalize_token(raw)
        if not token or token in _STOPWORDS:
            continue
        if len(token) >= 3 or token in _SHORT_TECH_TOKENS:
            tokens.add(token)
    return tokens


def _passes_relevance_gate(query_tokens: set[str], result: RetrievedSkill) -> bool:
    """Keep strong semantic hits and require lexical corroboration for weaker ones."""
    if result.final_rank <= _STRONG_MATCH_MAX_RANK:
        return True
    if result.final_rank > _WEAK_MATCH_MAX_RANK:
        return False
    activation_tokens = _content_tokens(result.skill.activation)
    return len(query_tokens & activation_tokens) >= 2


@dataclass
class RetrievalDiagnostics:
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    candidate_hits: int = 0
    final_hits: int = 0
    lexical_prefilter_skipped: bool = False


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config
        self.last_diagnostics = RetrievalDiagnostics()

    def _passes_lexical_prefilter(self, query_tokens: set[str]) -> bool:
        if len(query_tokens) < 3:
            return True
        for skill in self.store.list_skills(limit=None):
            if skill.maturity is Maturity.CANDIDATE:
                continue
            if query_tokens & _content_tokens(skill.activation):
                return True
        return False

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedSkill]:
        """Return top-k skills relevant to `query`, ranked."""
        start = time.perf_counter()
        self.last_diagnostics = RetrievalDiagnostics()
        if not query or not query.strip():
            self.last_diagnostics.total_ms = (time.perf_counter() - start) * 1000
            return []

        # Fast path: if the store has no skills yet, don't pay the
        # ~170ms cost of loading fastembed. This is the common case
        # for the first few Claude Code sessions after `mm init`,
        # before any skills have been extracted.
        if self.store.count_skills() == 0:
            self.last_diagnostics.total_ms = (time.perf_counter() - start) * 1000
            return []

        query_tokens = _content_tokens(query)
        if not self._passes_lexical_prefilter(query_tokens):
            self.last_diagnostics.lexical_prefilter_skipped = True
            self.last_diagnostics.total_ms = (time.perf_counter() - start) * 1000
            return []

        k = top_k if top_k is not None else self.config.retrieval_top_k

        embed_start = time.perf_counter()
        query_emb = self.embedder.embed_one(query)
        self.last_diagnostics.embed_ms = (time.perf_counter() - embed_start) * 1000

        # fetch a larger candidate pool, then rerank
        search_start = time.perf_counter()
        hits = self.store.search_skills_by_embedding(
            query_emb,
            top_k=max(k * 3, 10),
            scope=None,  # project + global both visible
        )
        self.last_diagnostics.search_ms = (time.perf_counter() - search_start) * 1000
        self.last_diagnostics.candidate_hits = len(hits)

        if not hits:
            self.last_diagnostics.total_ms = (time.perf_counter() - start) * 1000
            return []

        rerank_start = time.perf_counter()
        results: list[RetrievedSkill] = []
        for skill, distance in hits:
            if distance > 1.5:  # absurdly far
                continue
            if skill.maturity is Maturity.CANDIDATE:
                continue

            bonus = _MATURITY_BONUS.get(skill.maturity, 0.0) + (0.05 * min(skill.score, 1.0))
            results.append(RetrievedSkill(skill=skill, distance=distance, score_bonus=bonus))

        results.sort(key=lambda r: r.final_rank)

        # hard floor on similarity: distance after bonus still has to be reasonable
        floor = self.config.retrieval_similarity_floor
        results = [r for r in results if r.final_rank <= (2.0 - floor * 2.0)]
        results = [r for r in results if _passes_relevance_gate(query_tokens, r)]
        final_results = results[:k]
        self.last_diagnostics.rerank_ms = (time.perf_counter() - rerank_start) * 1000
        self.last_diagnostics.final_hits = len(final_results)
        self.last_diagnostics.total_ms = (time.perf_counter() - start) * 1000

        return final_results

    def mark_activated(self, retrieved: list[RetrievedSkill]) -> None:
        """Bump invocation count and last-used timestamp for activated skills."""
        now = datetime.now(UTC)
        for r in retrieved:
            r.skill.invocations += 1
            r.skill.last_used_at = now
            # score and maturity get updated when the episode completes,
            # not here — we don't yet know the outcome.
            self.store.update_skill(r.skill)
