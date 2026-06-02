"""Evaluate retrieval quality across chunk-size candidates."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from langchain_chroma import Chroma

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CHROMA_COLLECTION_NAME, DATA_DIR
from src.evaluation import evaluate_retrieval_docs, get_eval_cases
from src import indexer
from src.pdf_loader import find_pdf_files
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.source_resolver import format_source_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune chunk size with retrieval evals")
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[800, 1200, 1500, 2000],
        help="Chunk sizes to evaluate.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.1,
        help="Chunk overlap ratio relative to chunk size.",
    )
    parser.add_argument(
        "--keep-indexes",
        action="store_true",
        help="Keep temporary Chroma indexes under /tmp for inspection.",
    )
    parser.add_argument(
        "--case-set",
        choices=["all", "legal", "blockchain"],
        default="blockchain",
        help="Evaluation case group to run.",
    )
    return parser.parse_args()


def build_eval_vectorstore(chunks, chunk_size: int) -> tuple[Chroma, Path]:
    persist_dir = Path(tempfile.gettempdir()) / f"ollama_rag_chunk_eval_{chunk_size}"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=indexer.create_embedding_function(),
        ids=indexer.make_chunk_ids(chunks),
        persist_directory=str(persist_dir),
        collection_name=f"{CHROMA_COLLECTION_NAME}_{chunk_size}",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore, persist_dir


def evaluate_chunk_size(
    pdf_paths: list[Path],
    chunk_size: int,
    overlap: int,
    case_set: str,
) -> dict:
    original_chunk_size = indexer.CHUNK_SIZE
    original_chunk_overlap = indexer.CHUNK_OVERLAP
    indexer.CHUNK_SIZE = chunk_size
    indexer.CHUNK_OVERLAP = overlap
    try:
        start = time.perf_counter()
        chunks = indexer.build_chunks(pdf_paths)
        build_seconds = time.perf_counter() - start

        vectorstore, persist_dir = build_eval_vectorstore(chunks, chunk_size)
        results = []
        cases = get_eval_cases(case_set)
        for case in cases:
            query_start = time.perf_counter()
            docs = retrieve_hybrid(vectorstore, chunks, case.query)
            query_seconds = time.perf_counter() - query_start
            joined_context = "\n".join(doc.page_content for doc in docs)
            source_text = format_source_pages(docs, case.query, joined_context)
            result = evaluate_retrieval_docs(docs, case)
            result["sources"] = source_text
            result["top_pages"] = [str(doc.metadata.get("page", "?")) for doc in docs[:3]]
            result["seconds"] = query_seconds
            results.append(result)

        page_hits = sum(result["ref_hit"] for result in results)
        term_hits = sum(result["term_hit"] for result in results)
        total = len(results)
        avg_query_seconds = sum(result["seconds"] for result in results) / total
        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": len(chunks),
            "build_seconds": build_seconds,
            "page_hits": page_hits,
            "term_hits": term_hits,
            "total": total,
            "avg_query_seconds": avg_query_seconds,
            "results": results,
            "persist_dir": persist_dir,
        }
    finally:
        indexer.CHUNK_SIZE = original_chunk_size
        indexer.CHUNK_OVERLAP = original_chunk_overlap


def print_report(evaluations: list[dict]) -> None:
    print("\n# Chunk Size Evaluation\n")
    print("| chunk | overlap | chunks | page hit | term hit | avg query | build |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for evaluation in evaluations:
        print(
            f"| {evaluation['chunk_size']} | {evaluation['overlap']} | "
            f"{evaluation['chunks']} | {evaluation['page_hits']}/{evaluation['total']} | "
            f"{evaluation['term_hits']}/{evaluation['total']} | "
            f"{evaluation['avg_query_seconds']:.2f}s | "
            f"{evaluation['build_seconds']:.2f}s |"
        )

    best = max(
        evaluations,
        key=lambda item: (
            item["page_hits"],
            item["term_hits"],
            -item["avg_query_seconds"],
            -item["chunks"],
        ),
    )
    print(
        f"\nRecommended: CHUNK_SIZE={best['chunk_size']}, "
        f"CHUNK_OVERLAP={best['overlap']}"
    )

    for evaluation in evaluations:
        print(f"\n## CHUNK_SIZE={evaluation['chunk_size']}")
        for result in evaluation["results"]:
            status = "OK" if result["both_hit"] else "MISS"
            print(
                f"- {status} {result['id']}: {result['sources']} "
                f"top={result['top_pages']} {result['seconds']:.2f}s"
            )


def main() -> None:
    args = parse_args()
    pdf_paths = find_pdf_files(DATA_DIR)
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {DATA_DIR}")

    evaluations = []
    for chunk_size in args.chunk_sizes:
        overlap = max(1, int(chunk_size * args.overlap_ratio))
        print(f"Evaluating chunk_size={chunk_size}, overlap={overlap}...")
        evaluation = evaluate_chunk_size(pdf_paths, chunk_size, overlap, args.case_set)
        evaluations.append(evaluation)
        if not args.keep_indexes:
            shutil.rmtree(evaluation["persist_dir"], ignore_errors=True)

    print_report(evaluations)


if __name__ == "__main__":
    main()
