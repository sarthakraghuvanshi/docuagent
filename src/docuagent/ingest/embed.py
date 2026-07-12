from docuagent.ingest.chunk import Chunk
from docuagent.providers.embeddings import get_embedder


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    texts = [chunk.text for chunk in chunks]
    return get_embedder().embed(texts)
