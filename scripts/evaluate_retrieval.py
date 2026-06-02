"""Evaluate retrieval accuracy and source selection without calling the LLM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.evaluation import (
    any_expected_ref_hit,
    evaluate_retrieval_docs,
    expected_ref_hit,
    get_eval_cases,
    latency_stats,
    unexpected_refs,
)
from src.indexer import create_or_load_vectorstore, load_chunks
from src.rag_service import has_multiple_sources
from src.retrieval.source_resolver import format_source_pages, select_source_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval/source quality against local PDF index.",
    )
    parser.add_argument(
        "--case-set",
        choices=["all", "legal", "blockchain"],
        default="all",
        help="Evaluation case group to run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat retrieval for latency. Quality is reported from the first run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.RETRIEVAL_TOP_K,
        help="Hybrid retrieval top_k.",
    )
    parser.add_argument(
        "--reranker",
        choices=["env", "on", "off"],
        default="env",
        help="Override USE_RERANKER for this run.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write machine-readable results.",
    )
    return parser.parse_args()


def set_reranker_mode(mode: str) -> None:
    """Optionally override reranker state after modules are imported."""
    if mode == "env":
        return

    enabled = mode == "on"
    import src.retrieval.reranker as reranker_module
    import src.retrieval.hybrid as retriever_module

    config.USE_RERANKER = enabled
    reranker_module.USE_RERANKER = enabled
    retriever_module.USE_RERANKER = enabled


def print_report(results: list[dict], timings: list[float], reranker_enabled: bool) -> None:
    stats = latency_stats(timings)
    total = len(results)
    ref_hits = sum(bool(result["ref_hit"]) for result in results)
    term_hits = sum(bool(result["term_hit"]) for result in results)
    both_hits = sum(bool(result["both_hit"]) for result in results)
    source_hits = sum(bool(result["source_ref_hit"]) for result in results)
    source_clean = sum(bool(result["source_clean"]) for result in results)

    print("\n# Retrieval / Source Evaluation\n")
    print(f"- cases: {total}")
    print(f"- reranker: {'on' if reranker_enabled else 'off'}")
    print(
        f"- latency: avg {stats['avg']:.2f}s, median {stats['median']:.2f}s, "
        f"p95 {stats['p95']:.2f}s"
    )
    print()
    print("| metric | hit |")
    print("|---|---:|")
    print(f"| expected source/page | {ref_hits}/{total} |")
    print(f"| selected source hit | {source_hits}/{total} |")
    print(f"| selected source clean | {source_clean}/{total} |")
    print(f"| expected terms | {term_hits}/{total} |")
    print(f"| both | {both_hits}/{total} |")

    print("\n| case | status | retrieval | source clean | terms | first run | sources |")
    print("|---|---|---:|---:|---:|---:|---|")
    for result in results:
        status = "OK" if result["both_hit"] else "MISS"
        print(
            f"| {result['id']} | {status} | {result['ref_hit']} | "
            f"{result['source_clean']} | {result['term_hit']} | {result['seconds']:.2f}s | "
            f"{result['sources']} |"
        )

    misses = [
        result
        for result in results
        if not result["both_hit"] or not result["source_clean"]
    ]
    if misses:
        print("\n## Miss Details")
        for result in misses:
            print(
                f"- {result['id']}: top={result['top_refs']} "
                f"extra_sources={result['source_extra_refs']}"
            )


def main() -> None:
    args = parse_args()
    set_reranker_mode(args.reranker)

    import src.retrieval.hybrid as retriever_module

    vectorstore = create_or_load_vectorstore(rebuild=False)
    chunks = load_chunks()
    cases = get_eval_cases(args.case_set)
    show_source_names = has_multiple_sources(chunks)

    results: list[dict] = []
    timings: list[float] = []
    for run_index in range(max(1, args.repeat)):
        for case in cases:
            start = time.perf_counter()
            docs = retriever_module.retrieve_hybrid(
                vectorstore,
                chunks,
                case.query,
                top_k=args.top_k,
            )
            seconds = time.perf_counter() - start
            timings.append(seconds)

            if run_index > 0:
                continue

            result = evaluate_retrieval_docs(docs, case)
            selected_docs = select_source_docs(docs, case.query)
            source_extra_refs = unexpected_refs(selected_docs, case.expected_refs)
            result["seconds"] = seconds
            result["source_ref_hit"] = expected_ref_hit(selected_docs, case.expected_refs)
            result["source_ref_overlap"] = any_expected_ref_hit(
                selected_docs,
                case.expected_refs,
            )
            result["source_extra_refs"] = source_extra_refs
            result["source_clean"] = result["source_ref_overlap"] and not source_extra_refs
            result["sources"] = format_source_pages(
                docs,
                case.query,
                show_source_names=show_source_names,
            )
            results.append(result)

    print_report(results, timings, config.USE_RERANKER)
    if args.json_out:
        payload = {
            "case_set": args.case_set,
            "repeat": args.repeat,
            "top_k": args.top_k,
            "reranker": config.USE_RERANKER,
            "latency": latency_stats(timings),
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
