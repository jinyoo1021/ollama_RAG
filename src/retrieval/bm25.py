"""Korean BM25 retrieval using kiwipiepy tokenization."""

from __future__ import annotations

from langchain_core.documents import Document

from config import RETRIEVAL_TOP_K
from src.retrieval.document_utils import (
    clone_document,
    normalize_scores,
    tokenize_for_evidence,
)


_KIWI = None


def get_kiwi():
    """Create Kiwi lazily so simple tests do not pay tokenizer startup cost."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi

        _KIWI = Kiwi()
    return _KIWI


def tokenize_korean(text: str) -> list[str]:
    """Tokenize Korean text for BM25, with a regex fallback."""
    try:
        return [
            token.form.lower()
            for token in get_kiwi().tokenize(text)
            if token.form.strip()
        ]
    except Exception:
        return list(tokenize_for_evidence(text))


def retrieve_bm25(
    chunks: list[Document],
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[Document]:
    """Return BM25 matches using Korean morphological tokenization."""
    if not chunks:
        return []

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize_korean(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize_korean(query)
    scores = list(bm25.get_scores(query_tokens))
    normalized_scores = normalize_scores(scores)
    ranked_indexes = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    docs: list[Document] = []
    for index in ranked_indexes:
        if scores[index] <= 0:
            continue
        doc = clone_document(chunks[index])
        doc.metadata = {
            **doc.metadata,
            "bm25_score": round(normalized_scores[index], 4),
            "score": round(normalized_scores[index], 4),
            "retrieval": "bm25",
        }
        docs.append(doc)
    return docs
