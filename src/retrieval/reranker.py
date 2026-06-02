"""Cross-encoder reranking that avoids FlagEmbedding tokenizer API drift."""

from __future__ import annotations

from langchain_core.documents import Document

from config import (
    RERANK_TOP_K,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    USE_RERANKER,
)
from src.retrieval.document_utils import clone_document


_RERANKER = None


class CrossEncoderReranker:
    """Small transformers-based reranker avoiding FlagEmbedding tokenizer API drift."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device if device in {"cuda", "mps", "cpu"} else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def compute_score(self, pairs: list[list[str]]) -> list[float]:
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                queries = [pair[0] for pair in batch]
                passages = [pair[1] for pair in batch]
                inputs = self.tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    batch_scores = logits[:, -1]
                scores.extend(batch_scores.detach().cpu().float().tolist())
        return scores


def get_reranker() -> CrossEncoderReranker:
    """Load the reranker lazily when USE_RERANKER=true."""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker(
            model_name=RERANKER_MODEL,
            device=RERANKER_DEVICE,
            batch_size=RERANKER_BATCH_SIZE,
        )
    return _RERANKER


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = RERANK_TOP_K,
) -> list[Document]:
    """Rerank candidate docs with a cross-encoder when enabled."""
    if not USE_RERANKER or not docs:
        return docs

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.compute_score(pairs)
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    reranked_docs: list[Document] = []
    for doc, rerank_score in ranked[:top_k]:
        reranked_doc = clone_document(doc)
        reranked_doc.metadata = {
            **reranked_doc.metadata,
            "rerank_score": round(float(rerank_score), 4),
        }
        reranked_docs.append(reranked_doc)
    return reranked_docs
