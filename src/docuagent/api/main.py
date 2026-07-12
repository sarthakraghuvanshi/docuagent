from fastapi import FastAPI

from docuagent.api.routers import health, ingest, query
from docuagent.storage.db import init_db

app = FastAPI(title="DocuAgent", version="0.1.0")

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
