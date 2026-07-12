from fastapi import FastAPI

from docuagent.api.routers import health
from docuagent.storage.db import init_db

app = FastAPI(title="DocuAgent", version="0.1.0")

app.include_router(health.router)
# ingest and query routers are added in Phase 1, once those modules exist.


@app.on_event("startup")
def on_startup() -> None:
    init_db()
