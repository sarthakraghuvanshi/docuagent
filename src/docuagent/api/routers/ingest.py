import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from redis import Redis
from rq import Queue, Retry
from sqlalchemy.orm import Session

from docuagent.config import settings
from docuagent.ingest.worker import process_document
from docuagent.storage.db import get_session
from docuagent.storage.models import Document

router = APIRouter(prefix="/ingest", tags=["ingest"])

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

_redis = Redis.from_url(settings.redis_url)
_queue = Queue("ingest", connection=_redis)


@router.post("")
async def ingest_document(
    file: UploadFile = File(...), session: Session = Depends(get_session)
) -> dict:
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    existing = session.query(Document).filter_by(content_hash=content_hash).one_or_none()
    if existing is not None:
        return {"doc_id": existing.id, "status": existing.status, "deduped": True}

    doc = Document(
        filename=file.filename or "unknown",
        content_hash=content_hash,
        mime_type=file.content_type or "application/octet-stream",
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    dest_path = DATA_DIR / f"{doc.id}_{doc.filename}"
    dest_path.write_bytes(content)

    _queue.enqueue(
        process_document,
        doc.id,
        str(dest_path),
        doc.mime_type,
        retry=Retry(max=3, interval=[10, 30, 60]),
        job_timeout=600,
    )

    return {"doc_id": doc.id, "status": doc.status, "deduped": False}


@router.get("/status/{doc_id}")
def get_status(doc_id: str, session: Session = Depends(get_session)) -> dict:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "error": doc.error,
    }
