from fastapi import APIRouter
from sqlalchemy import text

from docuagent.providers.llm import get_llm
from docuagent.storage.db import engine
from docuagent.storage.qdrant_client import get_qdrant_client

router = APIRouter()


@router.get("/health")
def health() -> dict:
    checks: dict[str, str] = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # boundary check against an external service
        checks["postgres"] = f"error: {exc}"

    try:
        get_qdrant_client().get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    try:
        checks["ollama"] = "ok" if get_llm().is_healthy() else "error: not responding"
    except Exception as exc:
        checks["ollama"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
