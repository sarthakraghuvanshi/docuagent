# DocuAgent — Technical Blueprint

**Agentic Document Intelligence Platform**
Multi-agent RAG with self-healing retrieval. Local-first, cloud-ready.

Version 1.0 · Author: Sarthak Raghuvanshi · Target: buildable by a solo engineer, then scalable to production.

---

## 0. How to read this document

This is the single source of truth you build from. It goes top-down: what the system *is*, why each technology was chosen, how the agents fit together, then a phase-by-phase roadmap where every phase produces something you can run and demo. The last sections give you the exact repo layout, the evaluation harness that produces the accuracy/latency numbers, and the migration path to GCP when you want the "Vertex AI / Kubernetes / 30k users" story to be real rather than aspirational.

Guiding principle: **everything runs on your laptop in Phase 1 with zero cloud spend.** Cloud is a swap-in at the end, not a prerequisite. Every component below has a local implementation and a documented production upgrade.

---

## 1. What DocuAgent is

DocuAgent ingests documents (PDFs, DOCX, HTML, Markdown, images with text) and answers natural-language questions over them with grounded, cited answers. What makes it "agentic" rather than a plain RAG endpoint is that a small team of specialized agents — coordinated by a LangGraph state machine — plan the query, retrieve, *judge their own retrieval quality*, and self-heal by re-planning or falling back when the first attempt is weak. Instead of one linear `retrieve → stuff → generate` call, it's a graph with loops, grading, and conditional recovery.

### Core capabilities

1. **Document ingestion pipeline** — parse → chunk → embed → index, with per-document metadata and provenance.
2. **Query planning** — decompose complex questions, rewrite ambiguous ones, decide retrieval strategy.
3. **Hybrid retrieval** — dense (vector) + sparse (keyword/BM25) with a reranking pass.
4. **Self-healing / corrective loop** — a grader scores retrieved context; low scores trigger query rewrite, broadened search, or fallback, up to a bounded number of retries.
5. **Grounded answer synthesis** — answers cite the exact chunks they used; the model is instructed to say "not found in the documents" rather than hallucinate.
6. **Evaluation harness** — an offline A/B pipeline that measures retrieval accuracy, answer faithfulness, and latency across configurations, producing the numbers that justify design choices.

### Non-goals (explicitly out of scope for v1)

Real-time collaborative editing, fine-grained multi-tenant billing, and a polished consumer UI. The API and a thin demo UI are enough; the value is the retrieval intelligence.

---

## 2. Technology choices (and why)

The brief was "best-practice, my call, local-first." Each choice below favors something you can `docker compose up` today and still defend in a system-design interview.

| Concern | Local-first choice | Why | Production upgrade |
|---|---|---|---|
| Language | Python 3.11+ | Ecosystem for LLM/RAG is Python-native | same |
| Agent orchestration | **LangGraph** | Explicit stateful graphs with cycles — exactly what self-healing loops need; better than raw chains for conditional/looping control flow | same (LangGraph scales fine) |
| LLM (generation) | **Ollama** running Llama 3.1 8B or Qwen2.5 7B | Free, fully local, OpenAI-compatible API so you can swap providers with one env var | Vertex AI Gemini, or OpenAI/Anthropic API |
| Embeddings | **`BAAI/bge-small-en-v1.5`** (or `nomic-embed-text`) via sentence-transformers | Strong quality-to-size ratio, runs on CPU, and is fine-tunable — needed for the A/B eval story | Vertex AI embeddings or a fine-tuned model served on GPU |
| Vector store | **Qdrant** (Docker) | Native hybrid search, payload filtering, great local DX, horizontally scalable later | Qdrant Cloud / GKE, or Vertex AI Vector Search |
| Keyword/sparse | Qdrant sparse vectors **or** BM25 via the same index | Enables true hybrid retrieval without a second system | OpenSearch/Elasticsearch |
| Reranker | **`BAAI/bge-reranker-base`** cross-encoder | Big precision gain over pure vector similarity; small enough for CPU | GPU-served reranker or Cohere Rerank |
| Document parsing | **Docling** or **unstructured** + PyMuPDF | Handles PDF layout, tables, DOCX, HTML; PyMuPDF for fast text PDFs | same, plus Document AI for scanned docs |
| API layer | **FastAPI** | Async, typed, auto OpenAPI docs — you already use it | same |
| Relational / metadata | **Postgres** (Docker); DuckDB for local analytics | Stores documents, jobs, eval runs; DuckDB stands in for BigQuery-style analytics locally | Cloud SQL Postgres + BigQuery |
| Async ingestion workers | **Redis + RQ** (or Celery) | Ingestion is slow and bursty — must be off the request path | same on managed Redis |
| Observability / tracing | **Langfuse** (self-hosted Docker) or LangSmith | Trace every agent step, token, and latency; essential for debugging loops and for the eval dashboard | LangSmith or hosted Langfuse |
| Eval | **RAGAS** + a custom harness | Standard metrics (faithfulness, context precision/recall) plus your own A/B runner | same, wired into CI |
| Packaging | **Docker + docker-compose** | One command brings up the whole stack | Helm chart on GKE |
| Demo UI (optional) | **Streamlit** | Fastest path to a clickable demo; upload doc, ask, see citations + trace | Next.js if you want a real frontend |

**One rule that keeps this cheap and swappable:** all model calls (LLM + embeddings) go through a thin `providers/` abstraction with an OpenAI-compatible interface. Ollama, Vertex, and OpenAI all speak that shape, so moving from laptop to cloud is a config change, not a rewrite.

---

## 3. System architecture

### 3.1 High-level components

```
                        ┌─────────────────────────────────────────┐
                        │              FastAPI service             │
                        │   /ingest   /query   /eval   /health     │
                        └───────┬─────────────────────┬────────────┘
                                │                     │
                   enqueue job  │                     │  invoke graph
                                ▼                     ▼
                     ┌──────────────────┐   ┌────────────────────────┐
                     │  Ingestion       │   │  LangGraph query engine │
                     │  workers (RQ)    │   │  (multi-agent)          │
                     │  parse→chunk→    │   │  planner→retrieve→grade │
                     │  embed→index     │   │  →(loop)→synthesize     │
                     └───────┬──────────┘   └───────┬────────────────┘
                             │                      │
             ┌───────────────┼──────────────────────┼───────────────┐
             ▼               ▼                      ▼               ▼
        ┌─────────┐   ┌──────────────┐      ┌──────────────┐  ┌──────────┐
        │ Postgres│   │   Qdrant     │      │  Ollama /    │  │ Langfuse │
        │ metadata│   │ vectors +    │      │  embeddings  │  │ traces   │
        │ + jobs  │   │ sparse + rerank ctx │  + reranker  │  │          │
        └─────────┘   └──────────────┘      └──────────────┘  └──────────┘
```

### 3.2 The two pipelines

DocuAgent has two distinct paths that share storage:

- **Ingestion (write path, asynchronous):** triggered by `/ingest`, runs on a worker, is idempotent per document hash. Slow, batchy, retryable.
- **Query (read path, synchronous-ish):** triggered by `/query`, runs the LangGraph agent team, returns a cited answer plus a trace id. This is where the "agentic" intelligence lives.

Keeping them separate is deliberate: ingestion latency must never affect query latency, and each scales independently.

---

## 4. The agent team (LangGraph design)

The query engine is a LangGraph `StateGraph`. Think of it as a supervisor coordinating specialists, but implemented as an explicit graph with conditional edges and a bounded retry loop rather than free-form agent-to-agent chatter (which is harder to debug and to keep cheap).

### 4.1 Shared state

Every node reads and writes one typed state object:

```python
from typing import TypedDict, Annotated, Literal
from operator import add

class DocuState(TypedDict):
    question: str                     # original user question
    sub_questions: list[str]          # planner output
    strategy: Literal["semantic", "hybrid", "keyword"]
    queries: list[str]                # rewritten / expanded queries
    retrieved: list[dict]             # chunks: {id, text, score, source, page}
    grade: float                      # 0..1 context sufficiency
    grade_reason: str
    attempts: int                     # retry counter (bounds the loop)
    answer: str
    citations: list[dict]
    trace: Annotated[list[str], add]  # append-only step log
```

### 4.2 Nodes (the agents)

1. **Planner agent** — Reads `question`. Decides whether it's simple (one retrieval) or complex (decompose into `sub_questions`). Picks a `strategy` (semantic for conceptual questions, keyword/hybrid for exact-term or code/ID lookups). Produces rewritten `queries` (expansion, spelling/acronym normalization). This is a single LLM call with a structured-output schema.

2. **Retriever agent** — For each query: embed → Qdrant hybrid search (dense + sparse) with metadata filters (e.g. restrict to a document set) → merge/dedupe → cross-encoder **rerank** → keep top-k. Writes `retrieved`.

3. **Grader agent (the self-healing brain)** — Scores whether `retrieved` actually answers the question. Uses an LLM grader (Corrective-RAG / Self-RAG style) that returns a 0–1 `grade` plus a reason. This is the decision point for the loop.

4. **Synthesizer agent** — Generates the final answer *strictly from* `retrieved`, with inline citations mapping each claim to chunk ids. System prompt forbids using outside knowledge and requires an explicit "not found in the provided documents" when context is insufficient.

5. **(Optional) Reflection/critic agent** — A final pass that checks the drafted answer for faithfulness against the cited chunks (catches subtle hallucination) before returning.

### 4.3 Control flow with the self-healing loop

```
        ┌─────────┐
 START →│ planner │
        └────┬────┘
             ▼
        ┌───────────┐
        │ retriever │◀──────────────┐
        └────┬──────┘               │ (rewrite / broaden / fallback)
             ▼                       │
        ┌────────┐   grade < τ  &&   │
        │ grader │──  attempts<N  ───┘
        └────┬───┘
             │ grade ≥ τ  OR  attempts == N
             ▼
      ┌─────────────┐     ┌──────────┐
      │ synthesizer │ ──▶ │ critic   │ ──▶ END
      └─────────────┘     └──────────┘
```

**Conditional edge logic (the heart of "self-healing"):**

```python
def route_after_grade(state: DocuState) -> str:
    if state["grade"] >= GRADE_THRESHOLD:
        return "synthesize"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "synthesize_with_caveat"   # graceful degradation, not a crash
    return "recover"

# "recover" node picks an escalating fallback based on attempt #:
#   attempt 1 → rewrite/expand the query (planner re-invoked in "recovery" mode)
#   attempt 2 → switch strategy semantic↔hybrid, widen top-k, drop strict filters
#   attempt 3 → decompose further / retrieve per sub-question then merge
```

This bounded loop is what turns a brittle single-shot retrieval into a system that recovers from bad first attempts — and it's measurable: the "retrieval failure rate reduction" number comes directly from comparing failure rates with the loop on vs. off (see §7).

### 4.4 Why bounded, why a threshold

Two safety rails keep the agent from looping forever or burning tokens: `MAX_ATTEMPTS` (default 3) caps cost/latency, and `GRADE_THRESHOLD` (tune on your eval set, start ~0.7) defines "good enough." When it hits the cap without success, it degrades gracefully — synthesizes with an explicit low-confidence caveat rather than failing. Never crash on the user; always return the best available answer plus honesty about its confidence.

---

## 5. Ingestion pipeline (detailed)

The write path. Idempotent, resumable, observable.

### 5.1 Stages

1. **Intake & dedupe** — Compute a content hash of the uploaded file. If seen before, skip (idempotency). Record a `document` row (id, filename, hash, mime, status=`pending`).
2. **Parse** — Route by type: PyMuPDF for text PDFs (fast), Docling/unstructured for complex layout/tables/scanned, native readers for DOCX/HTML/MD. Output: normalized text blocks each carrying `{page, section, bbox?}` provenance.
3. **Chunk** — Structure-aware chunking: prefer semantic/heading boundaries, target ~500–800 tokens with ~15% overlap. Never split mid-table. Attach metadata (`doc_id`, `page`, `section`, `chunk_index`).
4. **Embed** — Batch chunks through the embedding provider. Also compute sparse representations for hybrid search.
5. **Index** — Upsert into Qdrant with the vector, sparse vector, and full payload (text + metadata). Mark `document.status = indexed`.
6. **Record** — Write chunk count, timing, and model/version used (so re-embeds are traceable) to Postgres.

### 5.2 Idempotency & re-embedding

Every indexed chunk stores the embedding model name + version in its payload. When you fine-tune a new embedding model (Phase 4), you re-embed into a *new Qdrant collection* and switch the query path over atomically — never mutate in place. This makes the A/B comparison in §7 clean: two collections, same corpus, different embeddings.

### 5.3 Failure handling

Parsing is the flakiest stage. Workers retry with backoff; a document that fails parsing N times is marked `failed` with the error captured, and never silently disappears. The `/ingest/status/{doc_id}` endpoint exposes this so the caller always knows where a document is.

---

## 6. Retrieval pipeline (detailed)

The quality of the whole system is dominated by retrieval quality, so this gets the most engineering care.

1. **Query embedding + sparse encoding** of each (possibly rewritten) query.
2. **Hybrid search in Qdrant** — combine dense similarity and sparse/keyword matches. Hybrid consistently beats either alone: dense catches paraphrase, sparse catches exact terms, IDs, and rare words.
3. **Metadata filtering** — restrict by document set, date, or type when the planner specifies it.
4. **Fusion & dedupe** — merge results across sub-queries (Reciprocal Rank Fusion), drop near-duplicate chunks.
5. **Reranking** — a cross-encoder rescoring of the top ~30 candidates down to the top ~5–8. This is the single highest-leverage precision improvement and cheap to add.
6. **Context assembly** — order by rerank score, trim to the model's context budget, keep citation metadata attached.

The retriever exposes knobs the recovery node manipulates: `top_k`, `strategy`, `filters`, `rerank_n`. That's what makes escalating fallback possible without new code paths.

---

## 7. Evaluation & the A/B pipeline

This section is what turns claims like "improved accuracy 21%, reduced latency 30%, cut failure rate 35%" into reproducible measurements rather than resume adjectives. Build the harness *early* (Phase 2) so every later change is measured, not guessed.

### 7.1 Build a gold eval set

Create 60–150 question/answer pairs over a fixed corpus, each labeled with the chunk(s) that contain the answer (the "relevant set"). Mix easy factoid, multi-hop, exact-term, and "answer not in corpus" negatives. This is a one-time investment that pays off forever. Store it as a versioned JSONL in the repo.

### 7.2 Metrics

- **Retrieval:** Precision@k, Recall@k, MRR, and **context precision/recall** (via RAGAS).
- **Answer:** **faithfulness** (is every claim supported by retrieved context?), answer relevancy, and exact/semantic correctness vs. gold answers.
- **Failure rate:** fraction of questions where the answer is wrong *or* the system says "not found" when the answer was actually present. This is the number the self-healing loop is designed to move.
- **Latency:** p50/p95 end-to-end, plus per-node timing from Langfuse traces.
- **Cost:** tokens per query (matters once you're on a paid API).

### 7.3 The A/B runner

A single script runs the whole eval set through two configurations and diffs them:

```
python -m docuagent.eval.run \
    --dataset data/eval/gold_v1.jsonl \
    --config-a configs/baseline.yaml \      # e.g. no rerank, no self-heal loop
    --config-b configs/full.yaml \          # rerank + self-heal + fine-tuned embeddings
    --out reports/ab_2026_07.json
```

It emits a table of every metric for A and B and the delta. **This is how you generate each headline number:**

- *Failure-rate reduction* = failure_rate(loop **off**) − failure_rate(loop **on**), both on the same gold set.
- *Accuracy improvement from fine-tuning* = context-recall(base embeddings) vs. (fine-tuned embeddings), same corpus, two Qdrant collections.
- *Latency reduction* = p95(naive: large-k, no rerank, big context) vs. (rerank to small-k, tighter context) — reranking lets you send the LLM fewer, better chunks, which is often a net latency win despite the extra rerank step.

Run it in CI on every PR so regressions are caught. That is the difference between a demo and a platform.

### 7.4 Embedding fine-tuning (Phase 4, optional but high-value)

Using the gold set, generate positive/hard-negative pairs and fine-tune `bge-small` with a contrastive objective (sentence-transformers `MultipleNegativesRankingLoss`). Evaluate the fine-tuned model against the base with the exact A/B runner above. Only ship it if the delta is real on held-out questions. This is a legitimate, defensible "improved retrieval accuracy by X%" story because you measured it.

---

## 8. API surface

Minimal, typed, FastAPI. Enough to drive the demo and the eval harness.

| Method | Route | Purpose |
|---|---|---|
| POST | `/ingest` | Upload a document; returns `doc_id`, enqueues async processing |
| GET | `/ingest/status/{doc_id}` | pending / parsing / indexed / failed |
| POST | `/query` | `{question, doc_filter?}` → `{answer, citations[], confidence, trace_id, attempts}` |
| POST | `/eval/run` | Kick off an A/B eval run (async) |
| GET | `/health` | Liveness/readiness (checks Qdrant, Postgres, Ollama) |

`/query` always returns citations and the `attempts` count, so the self-healing behavior is visible to the caller and in the demo UI.

---

## 9. Repository layout

```
docuagent/
├── docker-compose.yml           # qdrant, postgres, redis, langfuse, ollama, api, worker
├── pyproject.toml
├── .env.example                 # provider keys/urls, thresholds, model names
├── README.md
├── configs/
│   ├── baseline.yaml            # A/B config A
│   └── full.yaml                # A/B config B
├── src/docuagent/
│   ├── api/                     # FastAPI app, routers, schemas
│   ├── providers/               # OpenAI-compatible LLM + embedding + rerank adapters
│   │   ├── llm.py               # Ollama / Vertex / OpenAI behind one interface
│   │   ├── embeddings.py
│   │   └── rerank.py
│   ├── ingest/
│   │   ├── parse.py  chunk.py  embed.py  index.py  worker.py
│   ├── retrieve/
│   │   ├── hybrid.py  fusion.py  rerank.py
│   ├── graph/                   # the LangGraph agent team
│   │   ├── state.py             # DocuState
│   │   ├── nodes/               # planner.py retriever.py grader.py synthesizer.py critic.py recover.py
│   │   └── build.py             # StateGraph wiring + conditional edges
│   ├── eval/
│   │   ├── run.py               # A/B runner
│   │   ├── metrics.py           # RAGAS wrappers + custom failure-rate
│   │   └── finetune_embeddings.py
│   ├── storage/                 # Postgres models, Qdrant client
│   └── config.py                # pydantic settings (thresholds, k, model names)
├── data/eval/gold_v1.jsonl      # versioned gold eval set
├── ui/streamlit_app.py          # demo: upload, ask, see citations + trace + attempts
├── tests/                       # unit + a small integration test over a fixture corpus
└── infra/
    └── helm/                    # Phase 5 GKE chart (empty until cloud phase)
```

---

## 10. Phased build roadmap

Each phase ends with something runnable and demoable. Do them in order; resist the urge to jump to fine-tuning before the eval harness exists.

### Phase 0 — Skeleton & infra (½–1 day)
`docker-compose` with Qdrant + Postgres + Redis + Ollama + Langfuse. FastAPI `/health` green against all of them. `providers/` abstraction with Ollama wired in. **Done when:** `docker compose up` and `curl /health` returns all-green.

### Phase 1 — Ingestion + naive RAG end-to-end (2–3 days)
Parse → chunk → embed → index working via a worker. A *linear* (non-agentic) `/query` that retrieves top-k and answers with citations. No loop, no rerank yet. **Done when:** you can upload a PDF and get a cited answer. This is your first demo.

### Phase 2 — Evaluation harness (1–2 days)
Build the gold eval set and the A/B runner with RAGAS + failure-rate metrics. Baseline the Phase-1 system. **Done when:** `eval/run.py` prints a metrics table. *Do not skip this — everything after is measured against it.*

### Phase 3 — Hybrid retrieval + reranking (2 days)
Add sparse vectors + RRF fusion + cross-encoder rerank. Re-run eval, record the precision/latency delta. **Done when:** the A/B report shows rerank improving precision (it will).

### Phase 4 — Agentic self-healing graph (3–4 days)
Convert the linear query path to the LangGraph team: planner, grader, bounded recovery loop, critic. Wire Langfuse tracing. Re-run eval with loop off vs. on to get the **failure-rate reduction** number. **Done when:** traces show the loop recovering on hard questions and the failure-rate delta is real.

### Phase 5 — Fine-tuned embeddings (2–3 days, optional)
Generate pairs from the gold set, fine-tune `bge-small`, re-embed into a new collection, A/B it. Ship only if the held-out delta is positive. Produces the **accuracy-improvement** number.

### Phase 6 — Demo UI + polish (1–2 days)
Streamlit app: upload, ask, show answer + citations + confidence + attempt count + a link to the Langfuse trace. This is what you screen-record for the portfolio.

### Phase 7 — Cloud (when you want the "Vertex AI / K8s / 30k users" story) (variable)
See §11. Swap providers to Vertex, containerize to GKE with the Helm chart, load-test to substantiate scale claims.

---

## 11. Cloud / GCP migration path

Because of the `providers/` abstraction and Docker packaging, going to GCP is additive, not a rewrite. The mapping:

| Local component | GCP production equivalent |
|---|---|
| Ollama LLM | Vertex AI (Gemini) via the same OpenAI-compatible adapter |
| sentence-transformers embeddings | Vertex AI embeddings, or your fine-tuned model on a Vertex endpoint / GPU node |
| Qdrant (Docker) | Qdrant on GKE, Qdrant Cloud, or Vertex AI Vector Search |
| Postgres (Docker) | Cloud SQL for Postgres |
| DuckDB analytics | BigQuery (ingestion metrics, eval history, usage analytics) |
| Redis (Docker) | Memorystore |
| docker-compose | GKE via Helm chart in `infra/helm/` |
| Langfuse (Docker) | Hosted Langfuse or LangSmith |
| manual runs | Cloud Build CI/CD: test → eval-gate → build → deploy |

**Substantiating the scale claims honestly:** "supports 30k users" and "reduced deployment time" become real when you (a) put the stateless API behind an HPA on GKE, (b) load-test with Locust/k6 and record the throughput at target p95, and (c) wire Cloud Build so a merge auto-deploys. Do this only after Phases 1–6; the local system is where the actual engineering learning is.

**CI/CD eval gate (the detail that impresses):** the pipeline runs the A/B eval harness and *fails the deploy if retrieval metrics regress beyond a threshold*. That closes the loop between the eval work in §7 and production — quality is enforced, not hoped for.

---

## 12. Risks & the honest hard parts

- **Retrieval quality is the whole game.** Most of your time should go into chunking, hybrid search, and reranking — not agent choreography. A fancy graph over bad retrieval is still bad.
- **The self-healing loop can burn tokens/latency.** The bounded `MAX_ATTEMPTS` and grade threshold are non-negotiable guardrails. Watch p95, not just p50.
- **The grader is an LLM judging an LLM.** It's imperfect; calibrate the threshold on your gold set and spot-check its decisions in Langfuse traces.
- **Local models are weaker than frontier APIs.** For development that's fine; if answer quality on hard questions matters for the demo, flip the provider env var to a hosted model for the recording — the abstraction makes it one line.
- **The eval set is the foundation.** A small, careless gold set produces misleading numbers. Spend real effort labeling it; treat it as a code asset.

---

## 13. First week, concretely

1. Stand up `docker-compose` (Phase 0).
2. Get one PDF ingested and one cited answer back through a linear pipeline (Phase 1).
3. Hand-label 40–60 gold Q/A pairs over that PDF set and run the baseline eval (Phase 2).

At that point you have a working, *measured* RAG system — and everything after (rerank, the agent loop, fine-tuning, cloud) is an incremental, quantified improvement on a foundation that already runs. That sequencing is the difference between finishing this and stalling.

---

*End of blueprint. Build Phase 0 → 6 locally first; treat Phase 7 (GCP) as the productionization chapter once the intelligence works on your laptop.*
