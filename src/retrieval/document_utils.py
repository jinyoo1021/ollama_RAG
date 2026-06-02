"""Shared document helpers used across BM25, vector, and source resolution."""

from __future__ import annotations

import re

from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
LAW_ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조")


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to 0..1 while keeping equal-score lists stable."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def doc_key(doc: Document) -> tuple[str, str, int]:
    """Identify equivalent chunks across vector and BM25 results."""
    return (
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("page", "")),
        hash(doc.page_content),
    )


def clone_document(doc: Document) -> Document:
    """Copy LangChain documents across Pydantic versions."""
    if hasattr(doc, "model_copy"):
        return doc.model_copy(deep=True)
    return doc.copy(deep=True)


def merge_unique_documents(*doc_groups: list[Document]) -> list[Document]:
    """Merge document lists while preserving first occurrence order."""
    merged: list[Document] = []
    seen: set[tuple[str, str, int]] = set()
    for docs in doc_groups:
        for doc in docs:
            key = doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return merged


def tokenize_for_evidence(text: str) -> set[str]:
    """Extract lightweight Korean/ASCII terms for source-page selection."""
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 2
    }


def tokens_are_related(left: str, right: str) -> bool:
    """Treat simple Korean suffix variations as matching terms."""
    return left == right or left in right or right in left


def token_overlap_ratio(left_text: str, right_text: str) -> float:
    """Return a lightweight fuzzy token overlap ratio."""
    left_tokens = tokenize_for_evidence(left_text)
    right_tokens = tokenize_for_evidence(right_text)
    if not left_tokens:
        return 0.0

    matched = 0
    for left_token in left_tokens:
        if any(tokens_are_related(left_token, right_token) for right_token in right_tokens):
            matched += 1
    return matched / len(left_tokens)
