"""Measure retrieval, prompt-building, source-formatting, and optional LLM latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chat_manager import build_rag_messages, stream_without_inline_sources
from src.evaluation import get_eval_cases, latency_stats
from src.llm_client import stream_chat
from src.rag_service import has_multiple_sources, load_rag_resources
from src.retrieval.hybrid import format_context, retrieve_hybrid
from src.retrieval.source_resolver import format_source_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure local RAG latency.")
    parser.add_argument(
        "--case-set",
        choices=["all", "legal", "blockchain"],
        default="legal",
        help="Eval case group used as latency queries.",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Custom query. Can be passed multiple times. Overrides --case-set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of case-set queries. 0 means all.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each query.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup retrieval runs before measurement.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Also measure Ollama generation latency.",
    )
    return parser.parse_args()


def selected_queries(args: argparse.Namespace) -> list[str]:
    """Return latency queries from custom args or eval cases."""
    if args.query:
        return args.query
    queries = [case.query for case in get_eval_cases(args.case_set)]
    if args.limit > 0:
        return queries[: args.limit]
    return queries


def print_summary(samples: list[dict], resource_seconds: float, include_llm: bool) -> None:
    retrieval = latency_stats([sample["retrieval_seconds"] for sample in samples])
    prompt = latency_stats([sample["prompt_seconds"] for sample in samples])
    source = latency_stats([sample["source_seconds"] for sample in samples])
    total = latency_stats([sample["total_seconds"] for sample in samples])
    llm = latency_stats([sample["llm_seconds"] for sample in samples if sample["llm_seconds"] is not None])

    print("\n# Latency Measurement\n")
    print(f"- resource load: {resource_seconds:.2f}s")
    print(f"- samples: {len(samples)}")
    print(f"- LLM generation: {'included' if include_llm else 'not measured'}")
    print()
    print("| component | avg | median | p95 | min | max |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, stats in [
        ("retrieval", retrieval),
        ("prompt", prompt),
        ("source", source),
        ("llm", llm),
        ("total", total),
    ]:
        if label == "llm" and not include_llm:
            continue
        print(
            f"| {label} | {stats['avg']:.2f}s | {stats['median']:.2f}s | "
            f"{stats['p95']:.2f}s | {stats['min']:.2f}s | {stats['max']:.2f}s |"
        )

    print("\n| query | retrieval | llm | total | docs |")
    print("|---|---:|---:|---:|---:|")
    for sample in samples:
        llm_text = "-" if sample["llm_seconds"] is None else f"{sample['llm_seconds']:.2f}s"
        print(
            f"| {sample['query']} | {sample['retrieval_seconds']:.2f}s | "
            f"{llm_text} | {sample['total_seconds']:.2f}s | {sample['doc_count']} |"
        )

    errors = [sample for sample in samples if sample.get("llm_error")]
    if errors:
        print("\n## LLM Errors")
        for sample in errors:
            print(f"- {sample['query']}: {sample['llm_error']}")


def main() -> None:
    args = parse_args()
    queries = selected_queries(args)
    if not queries:
        raise SystemExit("No queries selected.")

    resource_start = time.perf_counter()
    resources = load_rag_resources(rebuild=False, build_if_missing=True)
    resource_seconds = time.perf_counter() - resource_start
    show_source_names = has_multiple_sources(resources.chunks, resources.pdf_files)

    for _ in range(max(0, args.warmup)):
        retrieve_hybrid(resources.vectorstore, resources.chunks, queries[0])

    samples: list[dict] = []
    for _ in range(max(1, args.repeat)):
        for query in queries:
            total_start = time.perf_counter()

            retrieval_start = time.perf_counter()
            docs = retrieve_hybrid(resources.vectorstore, resources.chunks, query)
            retrieval_seconds = time.perf_counter() - retrieval_start

            prompt_start = time.perf_counter()
            preliminary_sources = format_source_pages(
                docs,
                query,
                show_source_names=show_source_names,
            )
            messages = build_rag_messages(format_context(docs), query, preliminary_sources)
            prompt_seconds = time.perf_counter() - prompt_start

            llm_seconds = None
            llm_error = ""
            answer = ""
            if args.include_llm:
                llm_start = time.perf_counter()
                try:
                    answer = "".join(stream_without_inline_sources(stream_chat(messages)))
                except Exception as exc:
                    llm_error = str(exc)
                llm_seconds = time.perf_counter() - llm_start

            source_start = time.perf_counter()
            format_source_pages(
                docs,
                query,
                answer,
                show_source_names=show_source_names,
            )
            source_seconds = time.perf_counter() - source_start

            samples.append(
                {
                    "query": query,
                    "retrieval_seconds": retrieval_seconds,
                    "prompt_seconds": prompt_seconds,
                    "source_seconds": source_seconds,
                    "llm_seconds": llm_seconds,
                    "llm_error": llm_error,
                    "total_seconds": time.perf_counter() - total_start,
                    "doc_count": len(docs),
                }
            )

    print_summary(samples, resource_seconds, args.include_llm)


if __name__ == "__main__":
    main()
