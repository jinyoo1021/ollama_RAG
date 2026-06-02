"""Compare retrieval quality and latency with reranker on/off."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.evaluation import (
    EvalCase,
    evaluate_retrieval_docs,
    get_eval_cases,
    latency_stats,
)
from src.indexer import create_or_load_vectorstore, load_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reranker quality and speed")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each evaluation mode. Quality is measured from the first run.",
    )
    parser.add_argument(
        "--case-set",
        choices=["all", "legal", "blockchain"],
        default="legal",
        help="Evaluation case group to run.",
    )
    return parser.parse_args()


def set_reranker_enabled(enabled: bool) -> None:
    import src.retrieval.reranker as reranker_module
    import src.retrieval.hybrid as retriever_module

    config.USE_RERANKER = enabled
    reranker_module.USE_RERANKER = enabled
    retriever_module.USE_RERANKER = enabled


def evaluate_mode(enabled: bool, repeat: int, cases: list[EvalCase]) -> dict:
    import src.retrieval.hybrid as retriever_module

    set_reranker_enabled(enabled)
    vectorstore = create_or_load_vectorstore(rebuild=False)
    chunks = load_chunks()
    results = []
    all_seconds = []

    for run_index in range(repeat):
        for case in cases:
            start = time.perf_counter()
            docs = retriever_module.retrieve_hybrid(vectorstore, chunks, case.query)
            seconds = time.perf_counter() - start
            all_seconds.append(seconds)

            if run_index > 0:
                continue

            result = evaluate_retrieval_docs(docs, case)
            result["seconds"] = seconds
            results.append(result)

    stats = latency_stats(all_seconds)
    page_hits = sum(result["ref_hit"] for result in results)
    term_hits = sum(result["term_hit"] for result in results)
    both_hits = sum(result["both_hit"] for result in results)
    return {
        "enabled": enabled,
        "results": results,
        "page_hits": page_hits,
        "term_hits": term_hits,
        "both_hits": both_hits,
        "total": len(results),
        "avg_seconds": stats["avg"],
        "median_seconds": stats["median"],
        "first_seconds": all_seconds[0] if all_seconds else 0.0,
        "avg_after_first": statistics.mean(all_seconds[1:]) if len(all_seconds) > 1 else 0.0,
    }


def print_report(evaluations: list[dict]) -> None:
    print("\n# Reranker Comparison\n")
    print("| mode | page hit | term hit | both hit | avg | median | first | avg after first |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for evaluation in evaluations:
        mode = "on" if evaluation["enabled"] else "off"
        print(
            f"| {mode} | "
            f"{evaluation['page_hits']}/{evaluation['total']} | "
            f"{evaluation['term_hits']}/{evaluation['total']} | "
            f"{evaluation['both_hits']}/{evaluation['total']} | "
            f"{evaluation['avg_seconds']:.2f}s | "
            f"{evaluation['median_seconds']:.2f}s | "
            f"{evaluation['first_seconds']:.2f}s | "
            f"{evaluation['avg_after_first']:.2f}s |"
        )

    for evaluation in evaluations:
        mode = "on" if evaluation["enabled"] else "off"
        print(f"\n## reranker {mode}")
        for result in evaluation["results"]:
            status = "OK" if result["both_hit"] else "MISS"
            print(
                f"- {status} {result['id']}: source={result['ref_hit']} "
                f"term={result['term_hit']} {result['seconds']:.2f}s "
                f"top={result['top_refs']}"
            )


def main() -> None:
    args = parse_args()
    cases = get_eval_cases(args.case_set)
    evaluations = [
        evaluate_mode(enabled=False, repeat=args.repeat, cases=cases),
        evaluate_mode(enabled=True, repeat=args.repeat, cases=cases),
    ]
    print_report(evaluations)


if __name__ == "__main__":
    main()
