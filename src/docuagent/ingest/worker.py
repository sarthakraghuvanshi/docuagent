import traceback

import redis
from rq import Queue, Worker

from docuagent.config import settings
from docuagent.ingest.chunk import chunk_blocks
from docuagent.ingest.embed import embed_chunks
from docuagent.ingest.index import index_chunks
from docuagent.ingest.parse import parse_document
from docuagent.storage.db import SessionLocal
from docuagent.storage.models import Document

QUEUE_NAME = "ingest"


def process_document(doc_id: str, file_path: str, mime_type: str) -> None:
    session = SessionLocal()
    try:
        doc = session.get(Document, doc_id)
        if doc is None:
            return

        doc.status = "parsing"
        session.commit()

        blocks = parse_document(file_path, mime_type)
        chunks = chunk_blocks(blocks)
        vectors = embed_chunks(chunks)
        index_chunks(doc_id, chunks, vectors)

        doc.status = "indexed"
        doc.chunk_count = len(chunks)
        doc.error = None
        session.commit()
    except Exception as exc:  # ingestion boundary: never let a doc vanish silently
        session.rollback()
        doc = session.get(Document, doc_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = f"{exc}\n{traceback.format_exc()}"
            session.commit()
        raise
    finally:
        session.close()


def main() -> None:
    conn = redis.from_url(settings.redis_url)
    queue = Queue(QUEUE_NAME, connection=conn)
    Worker([queue], connection=conn).work()


if __name__ == "__main__":
    main()
