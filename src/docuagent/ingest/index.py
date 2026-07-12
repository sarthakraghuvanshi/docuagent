import uuid

from qdrant_client.models import PointStruct

from docuagent.config import settings
from docuagent.ingest.chunk import Chunk
from docuagent.storage.qdrant_client import ensure_collection, get_qdrant_client

# Fixed namespace so the same (doc_id, chunk_index) always hashes to the same
# point id -> re-indexing a document overwrites its old points instead of
# duplicating them.
_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))


def index_chunks(doc_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    ensure_collection()
    client = get_qdrant_client()

    points = [
        PointStruct(
            id=_point_id(doc_id, chunk.chunk_index),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "text": chunk.text,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
