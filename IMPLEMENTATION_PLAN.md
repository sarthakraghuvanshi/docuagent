# DocuAgent — Implementation Plan

Companion to `DocuAgent_Technical_Blueprint.md`. That document says *what* and *why*; this one says *do this, then this, then check this box*. Work top to bottom — each phase is gated by a "Done when" check before you move to the next one. Do not skip Phase 2 (eval harness) no matter how tempting the agent graph looks.

**Legend:** `[ ]` task · 🎯 done-when gate · 📁 files/dirs touched · ⏱ rough solo-engineer estimate

---

## Phase −1: Repo & tooling bootstrap (not in blueprint, do first) — ⏱ 1–2 hrs

This exists only in the filesystem right now, no git, no scaffold. Get the skeleton and version control in place before Phase 0.

- [ ] `git init`, create `.gitignore` (Python, Docker, `.env`, `__pycache__`, `*.pyc`, `.venv`, `data/raw/`, `reports/*.json`)
- [ ] Create the full directory tree from Blueprint §9 as empty stub dirs with `.gitkeep` / placeholder `__init__.py`
- [ ] `pyproject.toml` with Python 3.11+, and dependency groups: `api`, `ingest`, `graph`, `eval`, `dev` (pytest, ruff, mypy)
- [ ] `.env.example` — every env var referenced anywhere in this plan (provider URLs, model names, thresholds, DB creds) documented with a comment, even if empty
- [ ] `README.md` — one paragraph on what this is, `docker compose up` quickstart (fill in as phases land)
- [ ] Pre-commit or `ruff`/`black` config so style is consistent from commit 1
- [ ] First commit: "scaffold: repo skeleton"

🎯 **Done when:** `git log` shows one commit, `pip install -e .` (or `uv sync`) succeeds from a clean clone, directory layout matches Blueprint §9.

📁 `docuagent/`, `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`

---

## Phase 0 — Skeleton & infra — ⏱ ½–1 day

Get every backing service running locally and provable via `/health`.

- [ ] `docker-compose.yml`: `qdrant`, `postgres`, `redis`, `ollama`, `langfuse` (+ its own Postgres/ClickHouse deps per Langfuse's compose recipe)
- [ ] `ollama pull llama3.1:8b` (or `qwen2.5:7b`) — document exact model tag in `.env.example`
- [ ] `src/docuagent/config.py` — pydantic `Settings` reading all env vars (URLs, model names, `GRADE_THRESHOLD`, `MAX_ATTEMPTS`)
- [ ] `src/docuagent/providers/` — `llm.py`, `embeddings.py`, `rerank.py` stubs behind one OpenAI-compatible interface; wire only Ollama for now, leave Vertex/OpenAI as a documented but unimplemented branch
- [ ] `src/docuagent/api/` — FastAPI app skeleton with `GET /health` that pings Qdrant, Postgres, and Ollama and returns per-service status (not just "ok")
- [ ] `Dockerfile` for the API service (and worker, can share one image with different entrypoints)

🎯 **Done when:** `docker compose up` brings up all five services, `curl localhost:8000/health` returns 200 with every dependency green.

📁 `docker-compose.yml`, `Dockerfile`, `src/docuagent/config.py`, `src/docuagent/providers/*.py`, `src/docuagent/api/main.py`

---

## Phase 1 — Ingestion + naive RAG end-to-end — ⏱ 2–3 days

First real demo: upload a PDF, ask a question, get a cited answer. No agent graph yet — linear pipeline only.

**Ingestion (write path)**
- [ ] Postgres schema/migration: `document(id, filename, content_hash, mime, status, created_at, error)`
- [ ] `ingest/parse.py` — PyMuPDF for text PDFs; stub the Docling/unstructured branch for complex layouts (can be a TODO if time-boxed)
- [ ] `ingest/chunk.py` — structure-aware chunking, ~500–800 tokens, ~15% overlap, never split mid-table; attach `{doc_id, page, section, chunk_index}`
- [ ] `ingest/embed.py` — batch through `providers/embeddings.py` (`bge-small-en-v1.5` via sentence-transformers)
- [ ] `ingest/index.py` — upsert into a Qdrant collection with vector + full payload
- [ ] `ingest/worker.py` — RQ worker consuming an ingest queue; content-hash dedupe for idempotency
- [ ] `POST /ingest` — accepts file upload, writes `document` row (`status=pending`), enqueues job, returns `doc_id`
- [ ] `GET /ingest/status/{doc_id}` — reads `document.status`

**Query (read path, linear only)**
- [ ] `retrieve/hybrid.py` — dense-only search against Qdrant for now (hybrid comes in Phase 3)
- [ ] `POST /query` — embed question → top-k retrieve → stuff into prompt → LLM call via `providers/llm.py` → return `{answer, citations[]}`
- [ ] Citation format locked in now: chunk id → `{doc_id, page, section, text_snippet}` — this shape is reused by every later phase

🎯 **Done when:** you upload a real PDF via `/ingest`, poll `/ingest/status` to `indexed`, call `/query` with a question about its contents, and get back an answer with correct page/section citations.

📁 `src/docuagent/ingest/*.py`, `src/docuagent/retrieve/hybrid.py`, `src/docuagent/api/routers/{ingest,query}.py`, `src/docuagent/storage/`

---

## Phase 2 — Evaluation harness — ⏱ 1–2 days — **do not skip**

Everything after this phase is a measured delta against this baseline, not a guess.

- [ ] Pick a small fixed corpus (3–8 PDFs you know well) — ingest it once, keep it stable
- [ ] Hand-label 40–60 Q/A pairs in `data/eval/gold_v1.jsonl`: question, gold answer, gold chunk id(s), plus a handful of deliberate "not in corpus" negatives
- [ ] `eval/metrics.py` — wrap RAGAS (faithfulness, context precision/recall, answer relevancy) + a custom `failure_rate` metric (wrong answer OR false "not found")
- [ ] `eval/run.py` — A/B runner: takes `--config-a`, `--config-b`, `--dataset`, runs both through `/query` (or in-process), emits a metrics table + JSON report
- [ ] `configs/baseline.yaml` — describes current Phase-1 system (no rerank, no hybrid, no loop)
- [ ] Run it once against Phase 1 itself (`baseline.yaml` vs itself) just to prove the harness works end-to-end and produces a report file
- [ ] Commit `gold_v1.jsonl` as a versioned artifact — treat edits to it like code review

🎯 **Done when:** `python -m docuagent.eval.run --dataset data/eval/gold_v1.jsonl --config-a configs/baseline.yaml --config-b configs/baseline.yaml --out reports/phase2_smoke.json` runs clean and prints a metrics table with non-zero, sane numbers.

📁 `data/eval/gold_v1.jsonl`, `src/docuagent/eval/*.py`, `configs/baseline.yaml`, `reports/`

---

## Phase 3 — Hybrid retrieval + reranking — ⏱ 2 days

- [ ] Add sparse vectors to the Qdrant collection (re-index the Phase-1 corpus, or add sparse alongside existing dense)
- [ ] `retrieve/hybrid.py` — combine dense + sparse search
- [ ] `retrieve/fusion.py` — Reciprocal Rank Fusion across sub-query results + near-duplicate dedupe
- [ ] `retrieve/rerank.py` — cross-encoder (`bge-reranker-base`) rescoring top ~30 → top ~5–8
- [ ] Expose `top_k`, `strategy`, `filters`, `rerank_n` as knobs on the retriever (these get reused by the recovery node in Phase 4 — don't hardcode them)
- [ ] `configs/full.yaml` — same as baseline but rerank + hybrid on
- [ ] Re-run `eval/run.py` with `baseline.yaml` vs `full.yaml`, save `reports/phase3_ab.json`

🎯 **Done when:** the A/B report shows precision/context-recall improving with rerank+hybrid on, and you can state the actual delta number (not "should be better" — the printed number).

📁 `src/docuagent/retrieve/{hybrid,fusion,rerank}.py`, `configs/full.yaml`, `reports/phase3_ab.json`

---

## Phase 4 — Agentic self-healing graph — ⏱ 3–4 days

Convert the linear `/query` into the LangGraph team from Blueprint §4.

- [ ] `graph/state.py` — `DocuState` TypedDict exactly as specced (question, sub_questions, strategy, queries, retrieved, grade, grade_reason, attempts, answer, citations, trace)
- [ ] `graph/nodes/planner.py` — structured-output LLM call: simple vs. complex, strategy pick, query rewrite/expansion
- [ ] `graph/nodes/retriever.py` — wraps Phase-3 hybrid+rerank retrieval per query, merges across sub-questions
- [ ] `graph/nodes/grader.py` — LLM grader returning `{grade: 0..1, reason}` against `GRADE_THRESHOLD`
- [ ] `graph/nodes/recover.py` — escalating fallback by attempt number (rewrite → widen/switch strategy → decompose further), per Blueprint §4.3
- [ ] `graph/nodes/synthesizer.py` — answer strictly from `retrieved`, inline citations, explicit "not found" instruction
- [ ] `graph/nodes/critic.py` — faithfulness pass over the draft before returning (optional but build it — it's cheap and it's your hallucination catch)
- [ ] `graph/build.py` — wire the `StateGraph` with the conditional edge (`route_after_grade`), `MAX_ATTEMPTS` and `GRADE_THRESHOLD` from config
- [ ] Langfuse tracing wired through every node call; `/query` returns `trace_id` and `attempts` per Blueprint §8
- [ ] Point `POST /query` at the graph instead of the Phase-1 linear path
- [ ] `configs/agentic.yaml` — full retrieval + loop on
- [ ] Re-run eval: loop-off (`full.yaml`) vs loop-on (`agentic.yaml`) → this produces your **failure-rate reduction** number

🎯 **Done when:** you can pull up a Langfuse trace showing the grader triggering a recovery and a second retrieval attempt succeeding on a hard gold-set question, and the A/B report shows a real failure-rate delta with the loop on vs off.

📁 `src/docuagent/graph/**`, `configs/agentic.yaml`, `reports/phase4_ab.json`

---

## Phase 5 — Fine-tuned embeddings (optional, high-value) — ⏱ 2–3 days

- [ ] Generate positive/hard-negative pairs from `gold_v1.jsonl` (positives = gold chunk per question; hard negatives = high-scoring wrong chunks from Phase-3 retrieval)
- [ ] `eval/finetune_embeddings.py` — fine-tune `bge-small` with `MultipleNegativesRankingLoss` (sentence-transformers)
- [ ] Re-embed the full corpus into a **new** Qdrant collection (never mutate the existing one in place — Blueprint §5.2)
- [ ] `configs/finetuned.yaml` pointing the query path at the new collection
- [ ] A/B: base embeddings vs fine-tuned, held-out portion of gold set not used in fine-tuning
- [ ] Ship the switch only if held-out delta is positive; otherwise document the negative result and keep the base model — that's still a legitimate, honest finding

🎯 **Done when:** you have a signed-off A/B number for context-recall(base) vs context-recall(fine-tuned) on held-out questions, and a clear go/no-go decision recorded.

📁 `src/docuagent/eval/finetune_embeddings.py`, `configs/finetuned.yaml`

---

## Phase 6 — Demo UI + polish — ⏱ 1–2 days

- [ ] `ui/streamlit_app.py` — upload doc → poll ingest status → ask question → render answer, citations (clickable to source page/section), confidence, attempt count, Langfuse trace link
- [ ] Tighten error states: failed ingest shown clearly, "not found in documents" rendered distinctly from a real answer
- [ ] `tests/` — unit tests for chunker, fusion/dedupe, grader routing logic; one small integration test over a fixture corpus (2–3 tiny docs) that exercises `/ingest` → `/query` end to end
- [ ] README pass: quickstart, architecture diagram (reuse Blueprint §3.1), how to run eval, screenshot/gif of the UI

🎯 **Done when:** you can screen-record a clean run — upload, ask a hard question, watch it self-heal, see citations and trace — with no manual intervention.

📁 `ui/streamlit_app.py`, `tests/**`, `README.md`

---

## Phase 7 — Cloud / production (GCP) — ⏱ variable, do last

Only start this once Phases 1–6 are solid locally — this is productionization, not where the intelligence gets built.

- [ ] Swap `providers/llm.py` and `providers/embeddings.py` to Vertex AI branches (env var flip, per the abstraction — verify it really is a one-line change)
- [ ] Qdrant → GKE-hosted or Qdrant Cloud; Postgres → Cloud SQL; Redis → Memorystore; Langfuse → hosted or self-host on GKE
- [ ] `infra/helm/` chart: API + worker deployments, HPA on the API, secrets via GCP Secret Manager
- [ ] Cloud Build pipeline: test → **eval-gate** (fail deploy if A/B metrics regress past threshold, Blueprint §11) → build → deploy
- [ ] Load test with Locust/k6 against the GKE deployment; record throughput at target p95 — this is what makes a "supports N users" claim real
- [ ] DuckDB analytics → BigQuery for ingestion metrics / eval history if you want the analytics story too

🎯 **Done when:** a merge to main auto-deploys via Cloud Build, the eval gate has actually blocked at least one bad change (test this deliberately), and you have a load-test report with real p95/throughput numbers.

📁 `infra/helm/**`, `cloudbuild.yaml`

---

## Cross-cutting, applies to every phase

- **Measure before you claim.** Every "improved X%" number must come out of `eval/run.py`, not intuition. If Phase 4 or 5 doesn't move the needle, say so — a documented null result is more credible than an inflated one.
- **Never mutate indexed data in place.** New embedding model or chunking strategy → new Qdrant collection, atomic cutover. This is what keeps A/B comparisons clean (Blueprint §5.2).
- **Watch p95, not just p50**, once the self-healing loop exists — it's the metric that catches a runaway retry path.
- **Commit `configs/*.yaml` and `reports/*.json` from each phase** — they're your evidence trail, not scratch files.

## Suggested calendar (solo engineer, focused time)

| Phase | Est. | Cumulative |
|---|---|---|
| −1 Bootstrap | 1–2 hrs | Day 1 |
| 0 Infra | ½–1 day | Day 1–2 |
| 1 Ingestion + naive RAG | 2–3 days | Day 2–5 |
| 2 Eval harness | 1–2 days | Day 5–7 |
| 3 Hybrid + rerank | 2 days | Day 7–9 |
| 4 Agentic graph | 3–4 days | Day 9–13 |
| 5 Fine-tuning (optional) | 2–3 days | Day 13–16 |
| 6 Demo UI + polish | 1–2 days | Day 16–18 |
| 7 Cloud | variable | after Day 18 |

~2.5–3.5 weeks of focused solo work to a fully agentic, measured, locally-running system (Phases 0–6); Phase 7 is a separate productionization sprint whenever you want the cloud-scale story.
