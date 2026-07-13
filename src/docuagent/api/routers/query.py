from pydantic import BaseModel

from fastapi import APIRouter

from docuagent.providers.llm import get_llm
from docuagent.retrieve.hybrid import RetrievedChunk, retrieve

router = APIRouter(prefix="/query", tags=["query"])

SYSTEM_PROMPT = (
    "You are DocuAgent, a document question-answering assistant. Answer the "
    "question using ONLY the provided context chunks. Every claim must be "
    "grounded in the context and reference its chunk id like [chunk_id]. If "
    "the context does not contain the answer, respond with exactly: "
    '"Not found in the provided documents." Never use outside knowledge.'
)

NOT_FOUND_ANSWER = "Not found in the provided documents."


def build_context_block(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{hit.chunk_id}] (doc={hit.doc_id}, page={hit.page}): {hit.text}" for hit in hits
    )


def synthesize_answer(question: str, hits: list[RetrievedChunk]) -> str:
    """The core answer-generation step, shared by the live /query route and
    the eval harness so both exercise identical logic."""
    if not hits:
        return NOT_FOUND_ANSWER
    context_block = build_context_block(hits)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
    return get_llm().complete(SYSTEM_PROMPT, user_prompt)


class QueryRequest(BaseModel):
    question: str
    doc_filter: list[str] | None = None
    top_k: int = 5


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    section: str | None
    text_snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@router.post("", response_model=QueryResponse)
def run_query(request: QueryRequest) -> QueryResponse:
    hits = retrieve(request.question, top_k=request.top_k, doc_filter=request.doc_filter)
    answer = synthesize_answer(request.question, hits)

    citations = [
        Citation(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            page=hit.page,
            section=hit.section,
            text_snippet=hit.text[:280],
        )
        for hit in hits
    ]
    return QueryResponse(answer=answer, citations=citations)
