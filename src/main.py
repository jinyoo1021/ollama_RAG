"""CLI entry point for the Phase 4 RAG pipeline."""

from __future__ import annotations

import argparse
import sys

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import REBUILD_INDEX  # noqa: E402
from src.rag_service import collect_pdfs, has_multiple_sources, normalize_final_answer  # noqa: E402


def index_command(rebuild: bool = REBUILD_INDEX) -> None:
    from src.indexer import build_index

    pdf_files = collect_pdfs()
    print(f"Indexing {len(pdf_files)} PDF file(s)...")
    _, chunks = build_index(pdf_files, rebuild=rebuild)
    if chunks:
        print(f"Done. Stored {len(chunks)} chunk(s) in Chroma.")
    else:
        print("Done. Reused existing Chroma index.")


def answer_question(
    query: str,
    rebuild: bool = REBUILD_INDEX,
    debug_retrieval: bool = False,
) -> None:
    from src.chat_manager import build_rag_messages, stream_without_inline_sources
    from src.indexer import build_index, create_or_load_vectorstore, load_chunks
    from src.llm_client import stream_chat
    from src.retrieval.hybrid import format_context, retrieve_hybrid
    from src.retrieval.debug import RetrievalDebugTrace, format_retrieval_debug
    from src.retrieval.source_resolver import format_source_pages

    if rebuild:
        pdf_files = collect_pdfs()
        print(f"Indexing {len(pdf_files)} PDF file(s)...")
        vectorstore, chunks = build_index(pdf_files, rebuild=True)
        print(f"Indexed {len(chunks)} chunk(s).")
    else:
        vectorstore = create_or_load_vectorstore(rebuild=False)
        chunks = load_chunks()
    show_source_names = has_multiple_sources(chunks)

    debug_trace = RetrievalDebugTrace() if debug_retrieval else None
    docs = retrieve_hybrid(vectorstore, chunks, query, debug_trace=debug_trace)
    if debug_trace is not None:
        print(format_retrieval_debug(debug_trace))
        print()
    if not docs:
        print("문서에서 해당 정보를 찾을 수 없습니다.")
        return

    preliminary_sources = format_source_pages(
        docs,
        query,
        show_source_names=show_source_names,
    )
    messages = build_rag_messages(format_context(docs), query, preliminary_sources)
    answer = ""
    for text in stream_without_inline_sources(stream_chat(messages)):
        print(text, end="", flush=True)
        answer += text
    final_answer = normalize_final_answer(answer)
    if final_answer != answer.strip():
        print(f"\n\n{final_answer}", end="", flush=True)
    sources = format_source_pages(
        docs,
        query,
        final_answer,
        show_source_names=show_source_names,
    )
    print(f"\n\n{sources}")


def chat_loop(
    rebuild: bool = REBUILD_INDEX,
    debug_retrieval: bool = False,
) -> None:
    from src.chat_manager import ChatManager, stream_without_inline_sources
    from src.indexer import build_index, create_or_load_vectorstore, load_chunks
    from src.llm_client import stream_chat
    from src.retrieval.hybrid import format_context, retrieve_hybrid
    from src.retrieval.debug import RetrievalDebugTrace, format_retrieval_debug
    from src.retrieval.source_resolver import format_source_pages

    if rebuild:
        pdf_files = collect_pdfs()
        print(f"Indexing {len(pdf_files)} PDF file(s)...")
        vectorstore, chunks = build_index(pdf_files, rebuild=True)
        print(f"Indexed {len(chunks)} chunk(s).")
    else:
        vectorstore = create_or_load_vectorstore(rebuild=False)
        chunks = load_chunks()
    show_source_names = has_multiple_sources(chunks)

    chat_manager = ChatManager()

    print("질문을 입력하세요. 종료하려면 exit 또는 quit를 입력하세요.")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "종료"}:
            break
        retrieval_query = chat_manager.build_retrieval_query(query)
        debug_trace = RetrievalDebugTrace() if debug_retrieval else None
        docs = retrieve_hybrid(
            vectorstore,
            chunks,
            retrieval_query,
            debug_trace=debug_trace,
        )
        if debug_trace is not None:
            print(format_retrieval_debug(debug_trace))
            print()
        if not docs:
            print("문서에서 해당 정보를 찾을 수 없습니다.")
            continue
        preliminary_sources = format_source_pages(
            docs,
            retrieval_query,
            show_source_names=show_source_names,
        )
        messages = chat_manager.build_messages(
            format_context(docs),
            query,
            preliminary_sources,
        )
        answer = ""
        for text in stream_without_inline_sources(stream_chat(messages)):
            print(text, end="", flush=True)
            answer += text
        final_answer = normalize_final_answer(answer)
        if final_answer != answer.strip():
            print(f"\n\n{final_answer}", end="", flush=True)
        chat_manager.add_turn(query, final_answer)
        sources = format_source_pages(
            docs,
            retrieval_query,
            final_answer,
            show_source_names=show_source_names,
        )
        print(f"\n\n{sources}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Korean PDF RAG chatbot")
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse the existing Chroma index instead of rebuilding it.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("index", help="Build the Chroma index from data/pdfs")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print vector, BM25, reranker, and final retrieval candidates.",
    )
    ask_parser.add_argument("query", help="Question to answer from the indexed PDFs")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Start a simple interactive QA loop",
    )
    chat_parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print retrieval candidates for each chat turn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuild = not args.reuse_index

    try:
        if args.command == "index":
            index_command(rebuild=rebuild)
        elif args.command == "ask":
            answer_question(
                args.query,
                rebuild=rebuild,
                debug_retrieval=args.debug_retrieval,
            )
        else:
            chat_loop(
                rebuild=rebuild,
                debug_retrieval=getattr(args, "debug_retrieval", False),
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
