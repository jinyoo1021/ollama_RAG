"""Hybrid retrieval orchestrator combining vector and BM25 candidates."""

from __future__ import annotations

from langchain_core.documents import Document

from config import (
    BM25_WEIGHT,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    USE_RERANKER,
    VECTOR_WEIGHT,
)
from src.retrieval.bm25 import retrieve_bm25
from src.retrieval.document_utils import (
    clone_document,
    doc_key,
    merge_unique_documents,
    normalize_scores,
)
from src.retrieval.legal_references import (
    normalize_reference_text,
    retrieve_cross_law_mentions,
    retrieve_legal_references,
    source_law_name,
)
from src.retrieval.reranker import rerank_documents
from src.retrieval.debug import RetrievalDebugTrace
from src.retrieval.source_resolver import score_evidence_doc


def expand_query(query: str) -> str:
    """Return the retrieval query.

    Kept as a hook for future generic query-rewrite logic. It intentionally avoids
    document- or domain-specific hard-coded expansions.
    """
    return query


def retrieve(
    vectorstore,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    *,
    expand: bool = True,
) -> list[Document]:
    """Return relevant chunks from vector search only."""
    retrieval_query = expand_query(query) if expand else query
    results = vectorstore.similarity_search_with_relevance_scores(
        retrieval_query,
        k=top_k,
    )
    docs: list[Document] = []
    for doc, score in results:
        doc.metadata = {**doc.metadata, "score": round(float(score), 4)}
        docs.append(doc)
    return sorted(
        docs,
        key=lambda doc: score_evidence_doc(doc, retrieval_query),
        reverse=True,
    )


def merge_hybrid_results(
    vector_docs: list[Document],
    bm25_docs: list[Document],
) -> list[Document]:
    """Merge vector and BM25 docs using weighted normalized scores."""
    merged: dict[tuple[str, str, int], Document] = {}
    vector_scores = normalize_scores(
        [float(doc.metadata.get("score") or 0.0) for doc in vector_docs]
    )

    for doc, normalized_score in zip(vector_docs, vector_scores):
        key = doc_key(doc)
        merged_doc = clone_document(doc)
        merged_doc.metadata = {
            **merged_doc.metadata,
            "vector_score": round(normalized_score, 4),
            "bm25_score": 0.0,
            "hybrid_score": round(normalized_score * VECTOR_WEIGHT, 4),
            "retrieval": "vector",
        }
        merged[key] = merged_doc

    for doc in bm25_docs:
        key = doc_key(doc)
        bm25_score = float(doc.metadata.get("bm25_score") or doc.metadata.get("score") or 0.0)
        if key in merged:
            existing = merged[key]
            vector_score = float(existing.metadata.get("vector_score") or 0.0)
            existing.metadata = {
                **existing.metadata,
                "bm25_score": round(bm25_score, 4),
                "hybrid_score": round(
                    vector_score * VECTOR_WEIGHT + bm25_score * BM25_WEIGHT,
                    4,
                ),
                "retrieval": "hybrid",
            }
        else:
            merged_doc = clone_document(doc)
            merged_doc.metadata = {
                **merged_doc.metadata,
                "vector_score": 0.0,
                "bm25_score": round(bm25_score, 4),
                "hybrid_score": round(bm25_score * BM25_WEIGHT, 4),
                "retrieval": "bm25",
            }
            merged[key] = merged_doc

    return sorted(
        merged.values(),
        key=lambda doc: (
            float(doc.metadata.get("hybrid_score") or 0.0),
            score_evidence_doc(doc),
        ),
        reverse=True,
    )


def retrieve_hybrid(
    vectorstore,
    chunks: list[Document],
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    *,
    debug_trace: RetrievalDebugTrace | None = None,
) -> list[Document]:
    """Return hybrid BM25 + vector retrieval results."""
    retrieval_query = expand_query(query)
    if debug_trace is not None:
        debug_trace.original_query = query
        debug_trace.retrieval_query = retrieval_query
        debug_trace.top_k = top_k
        debug_trace.rerank_top_k = RERANK_TOP_K
        debug_trace.use_reranker = USE_RERANKER

    vector_docs = retrieve(vectorstore, retrieval_query, top_k=top_k, expand=False)
    bm25_docs = retrieve_bm25(chunks, retrieval_query, top_k=top_k)
    if debug_trace is not None:
        debug_trace.record("vector", vector_docs)
        debug_trace.record("bm25", bm25_docs)

    candidates = merge_hybrid_results(vector_docs, bm25_docs)
    mention_docs = retrieve_cross_law_mentions(chunks, retrieval_query, candidates)
    reference_anchor_docs = merge_unique_documents(mention_docs, candidates)
    reference_docs = retrieve_legal_references(chunks, retrieval_query, reference_anchor_docs)
    if debug_trace is not None:
        debug_trace.record("merged", candidates)
        debug_trace.record("legal_mention", mention_docs)
        debug_trace.record("legal_reference", reference_docs)

    candidates = merge_unique_documents(mention_docs, candidates, reference_docs)
    reranked = rerank_documents(retrieval_query, candidates, top_k=RERANK_TOP_K)
    if debug_trace is not None:
        debug_trace.record("reranked", reranked)

    if mention_docs:
        mentioned_source_laws = {
            normalize_reference_text(str(doc.metadata.get("source_law", "")))
            for doc in mention_docs
            if doc.metadata.get("source_law")
        }
        same_source_candidates = [
            doc
            for doc in candidates[:top_k]
            if normalize_reference_text(
                source_law_name(str(doc.metadata.get("source", "")))
            )
            in mentioned_source_laws
        ]
        final_docs = merge_unique_documents(
            mention_docs,
            reference_docs,
            same_source_candidates,
        )
    elif USE_RERANKER:
        final_docs = merge_unique_documents(reranked, reference_docs)
    else:
        final_docs = merge_unique_documents(candidates[:top_k], reference_docs)
    if debug_trace is not None:
        debug_trace.record("final", final_docs)
    return final_docs


def format_context(docs: list[Document]) -> str:
    """Format retrieved chunks with source and page references."""
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("score", "?")
        retrieval = doc.metadata.get("retrieval", "")
        note = ""
        if retrieval == "legal_mention":
            source_law = doc.metadata.get("source_law", "해당 법")
            mentioned_law = doc.metadata.get("mentioned_law", "다른 법")
            note = (
                f"\n[검색 의도] {source_law} 문서 안에서 "
                f"{mentioned_law}을/를 언급하거나 참조하는 조항입니다."
            )
        elif retrieval == "legal_reference":
            referenced_law = doc.metadata.get("referenced_law", "참조 법률")
            referenced_article = doc.metadata.get("referenced_article", "")
            note = (
                f"\n[검색 의도] 앞선 문서가 참조한 {referenced_law} "
                f"제{referenced_article}조 원문입니다."
            )
        parts.append(
            f"[문서 {index} | source={source} | p.{page} | score={score}]{note}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)
