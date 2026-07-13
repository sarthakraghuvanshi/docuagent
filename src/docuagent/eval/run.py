"""A/B evaluation runner. See Blueprint §7.3.

    python -m docuagent.eval.run \\
        --dataset data/eval/gold_v1.jsonl \\
        --config-a configs/baseline.yaml \\
        --config-b configs/baseline.yaml \\
        --out reports/phase2_smoke.json
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from docuagent.api.routers.query import build_context_block, synthesize_answer
from docuagent.eval.metrics import (
    answer_similarity,
    is_failure,
    judge_faithfulness,
    reciprocal_rank,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from docuagent.retrieve.hybrid import retrieve


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_gold_set(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def answer_question(question: str, top_k: int) -> dict:
    start = time.perf_counter()
    hits = retrieve(question, top_k=top_k)
    answer = synthesize_answer(question, hits)
    context = build_context_block(hits) if hits else ""
    latency = time.perf_counter() - start
    return {
        "answer": answer,
        "context": context,
        "retrieved_ids": [h.chunk_id for h in hits],
        "latency": latency,
    }


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path) as f:
        rows = json.load(f)
    return {row["id"]: row for row in rows}


def save_checkpoint(checkpoint_path: Path, rows_by_id: dict[str, dict], gold_set: list[dict]) -> None:
    ordered = [rows_by_id[item["id"]] for item in gold_set if item["id"] in rows_by_id]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(ordered, f, indent=2)


def run_config(config: dict, gold_set: list[dict], checkpoint_path: Path) -> list[dict]:
    top_k = config.get("retrieval", {}).get("top_k", 5)
    done = load_checkpoint(checkpoint_path)
    if done:
        print(f"  resuming from checkpoint: {len(done)}/{len(gold_set)} questions already done")

    for item in gold_set:
        if item["id"] in done:
            print(f"  [{item['id']}] already done, skipping")
            continue

        result = answer_question(item["question"], top_k)
        similarity = answer_similarity(result["answer"], item["gold_answer"])
        faithful = judge_faithfulness(result["context"], result["answer"])
        failure = is_failure(item["type"], result["answer"], similarity)

        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "system_answer": result["answer"],
            "retrieved_ids": result["retrieved_ids"],
            "gold_chunk_ids": item["gold_chunk_ids"],
            "retrieval_recall": retrieval_recall_at_k(result["retrieved_ids"], item["gold_chunk_ids"]),
            "retrieval_precision": retrieval_precision_at_k(
                result["retrieved_ids"], item["gold_chunk_ids"]
            ),
            "mrr": reciprocal_rank(result["retrieved_ids"], item["gold_chunk_ids"]),
            "answer_similarity": similarity,
            "faithful": faithful,
            "is_failure": failure,
            "latency_seconds": result["latency"],
        }
        done[item["id"]] = row
        save_checkpoint(checkpoint_path, done, gold_set)
        print(
            f"  [{row['id']}] recall={row['retrieval_recall']:.2f} "
            f"sim={similarity:.2f} faithful={faithful} failure={failure} "
            f"({result['latency']:.1f}s)"
        )

    return [done[item["id"]] for item in gold_set]


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    latencies = sorted(r["latency_seconds"] for r in results)
    return {
        "n_questions": n,
        "mean_retrieval_recall": sum(r["retrieval_recall"] for r in results) / n,
        "mean_retrieval_precision": sum(r["retrieval_precision"] for r in results) / n,
        "mean_mrr": sum(r["mrr"] for r in results) / n,
        "mean_answer_similarity": sum(r["answer_similarity"] for r in results) / n,
        "faithfulness_rate": sum(1 for r in results if r["faithful"]) / n,
        "failure_rate": sum(1 for r in results if r["is_failure"]) / n,
        "p50_latency_seconds": latencies[n // 2],
        "p95_latency_seconds": latencies[min(n - 1, int(n * 0.95))],
    }


def print_table(name_a: str, agg_a: dict, name_b: str, agg_b: dict) -> None:
    rows = [
        ("n_questions", "{:.0f}"),
        ("mean_retrieval_recall", "{:.3f}"),
        ("mean_retrieval_precision", "{:.3f}"),
        ("mean_mrr", "{:.3f}"),
        ("mean_answer_similarity", "{:.3f}"),
        ("faithfulness_rate", "{:.3f}"),
        ("failure_rate", "{:.3f}"),
        ("p50_latency_seconds", "{:.2f}"),
        ("p95_latency_seconds", "{:.2f}"),
    ]
    header = f"{'metric':<26} {name_a:>14} {name_b:>14} {'delta':>10}"
    print(header)
    print("-" * len(header))
    for key, fmt in rows:
        a_val, b_val = agg_a[key], agg_b[key]
        print(f"{key:<26} {fmt.format(a_val):>14} {fmt.format(b_val):>14} {b_val - a_val:>+10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gold_set = load_gold_set(args.dataset)
    config_a = load_config(args.config_a)
    config_b = load_config(args.config_b)

    out_path = Path(args.out)
    checkpoint_a = out_path.with_suffix(".a.checkpoint.json")
    checkpoint_b = out_path.with_suffix(".b.checkpoint.json")

    print(f"Running config A ({args.config_a}) against {len(gold_set)} questions...")
    results_a = run_config(config_a, gold_set, checkpoint_a)
    agg_a = aggregate(results_a)

    if Path(args.config_a).resolve() == Path(args.config_b).resolve():
        print("\nConfig B is the same file as config A - reusing results instead of re-running.")
        results_b, agg_b = results_a, agg_a
    else:
        print(f"\nRunning config B ({args.config_b}) against {len(gold_set)} questions...")
        results_b = run_config(config_b, gold_set, checkpoint_b)
        agg_b = aggregate(results_b)

    print()
    print_table(config_a.get("name", "A"), agg_a, config_b.get("name", "B"), agg_b)

    report = {
        "dataset": args.dataset,
        "config_a": {"path": args.config_a, "aggregate": agg_a, "per_question": results_a},
        "config_b": {"path": args.config_b, "aggregate": agg_b, "per_question": results_b},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")

    checkpoint_a.unlink(missing_ok=True)
    checkpoint_b.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
