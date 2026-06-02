"""Human-readable retrieval debug traces."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from langchain_core.documents import Document

from src.retrieval.source_resolver import format_source_name, score_evidence_doc


@dataclass
class RetrievalDebugTrace:
    """Capture intermediate retrieval candidates for CLI debugging."""

    original_query: str = ""
    retrieval_query: str = ""
    top_k: int = 0
    rerank_top_k: int = 0
    use_reranker: bool = False
    stages: dict[str, list[Document]] = field(default_factory=dict)

    def record(self, stage: str, docs: list[Document]) -> None:
        self.stages[stage] = list(docs)


def _compact_text(text: str, max_chars: int = 120) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


def _format_score(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "-"


def _format_doc_line(index: int, doc: Document, query: str) -> str:
    metadata = doc.metadata or {}
    source = format_source_name(str(metadata.get("source", "")))
    page = metadata.get("page", "?")
    retrieval = metadata.get("retrieval", "-")
    score_parts = [
        f"score={_format_score(metadata.get('score'))}",
        f"vector={_format_score(metadata.get('vector_score'))}",
        f"bm25={_format_score(metadata.get('bm25_score'))}",
        f"hybrid={_format_score(metadata.get('hybrid_score'))}",
        f"rerank={_format_score(metadata.get('rerank_score'))}",
        f"evidence={score_evidence_doc(doc, query):.4f}",
    ]
    preview = _compact_text(doc.page_content)
    return (
        f"{index}. {source} p.{page} | retrieval={retrieval} | "
        f"{' | '.join(score_parts)}\n"
        f"   {preview}"
    )


def format_retrieval_debug(trace: RetrievalDebugTrace) -> str:
    """Format a retrieval trace for terminal output."""
    lines = [
        "[debug] retrieval",
        f"- query: {trace.original_query}",
        f"- retrieval_query: {trace.retrieval_query}",
        (
            f"- top_k: {trace.top_k}, rerank_top_k: {trace.rerank_top_k}, "
            f"use_reranker: {str(trace.use_reranker).lower()}"
        ),
    ]
    stage_titles = [
        ("vector", "vector candidates"),
        ("bm25", "bm25 candidates"),
        ("merged", "merged candidates"),
        ("legal_mention", "legal mention candidates"),
        ("legal_reference", "legal reference candidates"),
        ("reranked", "reranked candidates"),
        ("final", "final context docs"),
    ]
    for stage, title in stage_titles:
        docs = trace.stages.get(stage, [])
        lines.append(f"\n[{title}] {len(docs)}")
        if not docs:
            lines.append("-")
            continue
        for index, doc in enumerate(docs, start=1):
            lines.append(_format_doc_line(index, doc, trace.retrieval_query))
    return "\n".join(lines)
