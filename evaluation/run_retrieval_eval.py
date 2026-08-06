"""Retrieval evaluation for AlphaLens.

Measures whether retrieval surfaces the expected source for each question in
the benchmark dataset. This is not a test suite: it makes real API calls and
reports scores rather than passing or failing. Run it before and after any
change to chunking, embedding, or retrieval to see whether results moved.

Usage:
    python -m evaluation.run_retrieval_eval
    python -m evaluation.run_retrieval_eval --top-k 10
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from src.config import validate_config
from src.logging_config import setup_logging
from src.rag import StoreNotFoundError, ensure_store, retrieve

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _hit_rank(evidence, expect_type: str | None, expect_section: str | None) -> int | None:
    """Return the 1-based rank of the first matching chunk, or None."""
    for rank, item in enumerate(evidence, start=1):
        meta = item.metadata
        if expect_type and meta.get("source_type") != expect_type:
            continue
        if expect_section and meta.get("section") != expect_section:
            continue
        return rank
    return None


def run_evaluation(top_k: int) -> dict:
    """Run every dataset case and return a results summary."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    tickers = sorted({case["ticker"] for case in dataset["cases"]})
    for ticker in tickers:
        count, _ = ensure_store(ticker)
        if count == 0:
            print(f"WARNING: no store for {ticker}; its cases will fail.")

    results = []
    for case in dataset["cases"]:
        expect_type = case.get("expect_source_type")
        expect_section = case.get("expect_section")

        try:
            evidence = retrieve(case["ticker"], case["question"], n_results=top_k)
        except StoreNotFoundError:
            results.append({**case, "hit_rank": None, "error": "no_store"})
            continue

        rank = _hit_rank(evidence, expect_type, expect_section)
        top_score = evidence[0].score if evidence else None

        results.append({
            "id": case["id"],
            "ticker": case["ticker"],
            "expect_source_type": expect_type,
            "expect_section": expect_section,
            "hit_rank": rank,
            "top_score": top_score,
            "retrieved": [
                {
                    "source_type": e.metadata.get("source_type"),
                    "section": e.metadata.get("section"),
                    "score": e.score,
                }
                for e in evidence
            ],
        })

    answerable = [r for r in results if r["expect_source_type"] is not None]
    hits = [r for r in answerable if r["hit_rank"] is not None]
    top1 = [r for r in answerable if r["hit_rank"] == 1]
    mrr = sum(1 / r["hit_rank"] for r in hits) / len(answerable) if answerable else 0.0

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "dataset_version": dataset["version"],
        "answerable_cases": len(answerable),
        "recall_at_k": len(hits) / len(answerable) if answerable else 0.0,
        "precision_at_1": len(top1) / len(answerable) if answerable else 0.0,
        "mrr": mrr,
        "results": results,
    }


def main() -> None:
    setup_logging()
    validate_config()

    parser = argparse.ArgumentParser(description="Evaluate AlphaLens retrieval.")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question")
    parser.add_argument("--save", action="store_true", help="Write results to evaluation/results/")
    args = parser.parse_args()

    summary = run_evaluation(args.top_k)

    print(f"\n{'ID':<32} {'RANK':>5}  {'TOP SCORE':>10}")
    print("-" * 52)
    for r in summary["results"]:
        rank = r["hit_rank"] if r["hit_rank"] else ("n/a" if r["expect_source_type"] is None else "MISS")
        score = f"{r['top_score']:.3f}" if r.get("top_score") is not None else "-"
        print(f"{r['id']:<32} {rank!s:>5}  {score:>10}")

    print(f"\nAnswerable cases: {summary['answerable_cases']}")
    print(f"Recall@{summary['top_k']}:      {summary['recall_at_k']:.1%}")
    print(f"Precision@1:      {summary['precision_at_1']:.1%}")
    print(f"MRR:              {summary['mrr']:.3f}")

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RESULTS_DIR, f"retrieval-{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()