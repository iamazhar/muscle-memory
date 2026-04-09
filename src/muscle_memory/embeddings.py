"""Pluggable embedding backends.

The default is `fastembed` (local CPU, BGE-small, 384 dims) so
muscle-memory works out of the box without any API keys.

To swap providers set MM_EMBEDDER=openai or MM_EMBEDDER=voyage
(requires the corresponding optional extras).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from muscle_memory.config import Config


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into fixed-dimensional vectors."""

    dims: int

    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local, CPU-only embedder using fastembed.

    Lazy-loads the model on first use so that CLI commands that
    don't need embeddings stay snappy.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dims: int = 384):
        self.model_name = model_name
        self.dims = dims
        self._model: TextEmbedding | None = None

    def _load(self) -> None:
        if self._model is None:
            # imported lazily to keep startup fast
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        self._load()
        assert self._model is not None
        return [list(map(float, v)) for v in self._model.embed(list(texts))]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder:
    """OpenAI embeddings. Requires `openai` extra."""

    def __init__(self, model: str = "text-embedding-3-small", dims: int = 1536):
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "openai extra not installed. Run: uv add muscle-memory[openai]"
            ) from exc
        self.model = model
        self.dims = dims
        self._client: object | None = None

    def _load(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        self._load()
        assert self._client is not None
        resp = self._client.embeddings.create(  # type: ignore[attr-defined]
            model=self.model,
            input=list(texts),
        )
        return [list(d.embedding) for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VoyageEmbedder:
    """Voyage AI embeddings. Requires `voyage` extra."""

    def __init__(self, model: str = "voyage-3-lite", dims: int = 512):
        try:
            import voyageai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "voyage extra not installed. Run: uv add muscle-memory[voyage]"
            ) from exc
        self.model = model
        self.dims = dims
        self._client: object | None = None

    def _load(self) -> None:
        if self._client is None:
            import voyageai

            self._client = voyageai.Client()

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        self._load()
        assert self._client is not None
        resp = self._client.embed(  # type: ignore[attr-defined]
            list(texts), model=self.model, input_type="document"
        )
        return [list(v) for v in resp.embeddings]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_embedder(config: Config) -> Embedder:
    """Resolve the configured embedder."""
    provider = config.embedder.lower()
    if provider == "fastembed":
        return FastEmbedEmbedder(model_name=config.embedding_model, dims=config.embedding_dims)
    if provider == "openai":
        dims = config.embedding_dims if config.embedding_dims != 384 else 1536
        return OpenAIEmbedder(model=config.embedding_model, dims=dims)
    if provider == "voyage":
        dims = config.embedding_dims if config.embedding_dims != 384 else 512
        return VoyageEmbedder(model=config.embedding_model, dims=dims)
    raise ValueError(f"Unknown embedder: {provider}")
