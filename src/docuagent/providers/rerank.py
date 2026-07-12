"""Cross-encoder reranker. Stub for Phase 0/1 — implemented in Phase 3."""

from typing import Protocol


class RerankProvider(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def get_reranker() -> RerankProvider:
    raise NotImplementedError("Reranker lands in Phase 3 (see IMPLEMENTATION_PLAN.md).")
