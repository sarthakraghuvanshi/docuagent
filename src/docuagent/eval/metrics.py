"""Evaluation metrics: custom, RAGAS-inspired implementations rather than
the RAGAS library itself. Each metric mirrors what RAGAS measures (context
precision/recall, answer correctness, faithfulness) computed directly against
our own gold labels and local LLM. See docs/blueprint §7.2 for the metric
definitions this maps to.
"""

import numpy as np

from docuagent.providers.embeddings import get_embedder
from docuagent.providers.llm import get_llm

NOT_FOUND_PHRASE = "not found in the provided documents"
CORRECTNESS_THRESHOLD = 0.75


def is_refusal(answer: str) -> bool:
    return NOT_FOUND_PHRASE in answer.lower()


def retrieval_recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0
    hits = sum(1 for g in gold_ids if g in retrieved_ids)
    return hits / len(gold_ids)


def retrieval_precision_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    if not gold_ids:
        return 1.0
    hits = sum(1 for r in retrieved_ids if r in gold_ids)
    return hits / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0


def answer_similarity(system_answer: str, gold_answer: str) -> float:
    embedder = get_embedder()
    vec_a, vec_b = embedder.embed([system_answer, gold_answer])
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


FAITHFULNESS_PROMPT = (
    "You are grading whether an answer is faithful to its source context. "
    "Faithful means every factual claim in the answer is directly supported "
    "by the context, with no invented facts and no outside knowledge. "
    "Respond with exactly one word: YES if faithful, NO if not."
)


def judge_faithfulness(context: str, answer: str) -> bool:
    if is_refusal(answer):
        return True  # a refusal makes no claims, so it can't be unfaithful
    verdict = get_llm().complete(
        FAITHFULNESS_PROMPT,
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\nIs the answer faithful? YES or NO.",
        temperature=0.0,
    )
    return "yes" in verdict.strip().lower()[:10]


def is_failure(question_type: str, system_answer: str, similarity: float) -> bool:
    """Blueprint §7.2's failure_rate definition: wrong answer, or a false
    refusal when the answer was present, or (for negatives) a hallucinated
    answer instead of the expected refusal."""
    refused = is_refusal(system_answer)
    if question_type == "negative":
        return not refused
    if refused:
        return True
    return similarity < CORRECTNESS_THRESHOLD
