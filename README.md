# DocuAgent

Agentic document intelligence platform: ingest PDFs/DOCX/HTML/Markdown, ask natural-language
questions, get grounded, cited answers. A LangGraph multi-agent team plans the query, retrieves,
grades its own retrieval quality, and self-heals via bounded retry before answering. Local-first,
zero cloud spend by default — see `DocuAgent_Technical_Blueprint.md` for the full design and
`IMPLEMENTATION_PLAN.md` for the phase-by-phase build checklist.

## Quickstart

```bash
cp .env.example .env        # already done for local dev — edit if you change ports/models
docker compose up --build
curl localhost:8000/health  # should report postgres / qdrant / ollama all "ok"
```

Pull the local LLM once the `ollama` container is up:

```bash
docker compose exec ollama ollama pull llama3.1:8b
```

Then upload a document and ask a question:

```bash
curl -F "file=@/path/to/doc.pdf" localhost:8000/ingest
curl localhost:8000/ingest/status/<doc_id>          # wait for status: indexed
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Local development (without Docker for the API)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Status

Building phase by phase per `IMPLEMENTATION_PLAN.md`. Currently on **Phase 1 — Ingestion + naive
RAG end-to-end**.
