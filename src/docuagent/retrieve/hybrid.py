"""Retrieval: question -> ranked chunks.

Phase 1: dense (vector) search only. Phase 3 adds sparse vectors + RRF
fusion into this same function so the /query call site never has to change.
"""

from dataclasses import dataclass

from docuagent.config import settings
from docuagent.providers.embeddings import get_embedder
from docuagent.storage.qdrant_client import get_qdrant_client


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    section: str | None
    score: float


def retrieve(
    question: str, *, top_k: int | None = None, doc_filter: list[str] | None = None
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k
    query_vector = get_embedder().embed([question])[0]

    query_filter = None
    if doc_filter:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_filter))])

    client = get_qdrant_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            chunk_id=str(point.id),
            doc_id=point.payload["doc_id"],
            text=point.payload["text"],
            page=point.payload["page"],
            section=point.payload.get("section"),
            score=point.score,
        )
        for point in results
    ]
