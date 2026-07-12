"""Embedding provider behind one interface. Local sentence-transformers now,
Vertex/OpenAI embeddings are a config swap in Phase 7."""

from functools import lru_cache
from typing import Protocol

from docuagent.config import settings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingProvider:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings.embedding_model)
    raise NotImplementedError(
        f"Embedding provider '{settings.embedding_provider}' not wired yet (Phase 7)."
    )
