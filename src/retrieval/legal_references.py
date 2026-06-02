"""Cross-document legal article reference handling."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

from src.retrieval.document_utils import (
    TOKEN_PATTERN,
    clone_document,
    doc_key,
    token_overlap_ratio,
    tokenize_for_evidence,
    tokens_are_related,
)


QUOTED_LEGAL_REFERENCE_PATTERN = re.compile(
    r"「\s*([^」]+?)\s*」\s*제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?"
)
NAMED_LEGAL_REFERENCE_PATTERN = re.compile(
    r"([가-힣A-Za-z0-9ㆍ·\s]{2,40}?법)\s*제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?"
)
REFERENCE_ANCHOR_MIN_OVERLAP = 0.5
CROSS_LAW_MENTION_MIN_OVERLAP = 0.5
CROSS_LAW_MENTION_INTENT_PATTERN = re.compile(
    r"포함|들어\s*있|언급|참조|관련\s*조항|법령이?\s*존재|조항이?\s*존재"
)
CROSS_LAW_QUERY_SPLIT_PATTERN = re.compile(r"(.+?)(?:안에|내에|에서|중에)(.+)")


def normalize_reference_text(text: str) -> str:
    """Normalize Korean legal names across PDF text and decomposed filenames."""
    normalized = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", normalized)


def source_law_name(source: str) -> str:
    """Return a readable law name from a source PDF path."""
    stem = unicodedata.normalize("NFC", Path(source).stem)
    stem = re.sub(r"\([^)]*\)", "", stem)
    return stem.strip()


def extract_legal_references(text: str) -> list[tuple[str, str, str]]:
    """Extract cross-law article references such as 「근로자퇴직급여 보장법」 제12조제1항."""
    references: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for pattern in (QUOTED_LEGAL_REFERENCE_PATTERN, NAMED_LEGAL_REFERENCE_PATTERN):
        for match in pattern.finditer(text):
            law = re.sub(r"\s+", " ", match.group(1)).strip()
            article = match.group(2)
            paragraph = match.group(3) or ""
            key = (law, article, paragraph)
            if key not in seen:
                seen.add(key)
                references.append(key)

    return references


def candidate_supports_reference_expansion(query: str, doc: Document) -> bool:
    """Return True when a candidate is relevant enough to follow its references."""
    if doc.metadata.get("retrieval") == "legal_mention":
        return True
    return token_overlap_ratio(query, doc.page_content) >= REFERENCE_ANCHOR_MIN_OVERLAP


def law_name_is_query_related(query: str, law: str) -> bool:
    """Return True when the referenced law name shares a concrete query term."""
    query_tokens = tokenize_for_evidence(query)
    law_tokens = {
        token
        for token in tokenize_for_evidence(law)
        if token not in {"법", "법률", "보장법"}
    }
    return any(
        tokens_are_related(query_token, law_token)
        for query_token in query_tokens
        for law_token in law_tokens
    )


def query_has_cross_law_mention_intent(query: str) -> bool:
    """Return True when the query asks whether one law mentions another law."""
    return bool(CROSS_LAW_MENTION_INTENT_PATTERN.search(query))


def law_name_matches_query(query: str, law: str) -> bool:
    """Match a corpus law name against possibly spaced or slightly varied query text."""
    normalized_law = normalize_reference_text(law)
    normalized_query = normalize_reference_text(query)
    return (
        normalized_law in normalized_query
        or token_overlap_ratio(law, query) >= CROSS_LAW_MENTION_MIN_OVERLAP
    )


def law_name_matches_content(law: str, content: str) -> bool:
    """Return True when a law name appears in content."""
    normalized_law = normalize_reference_text(law)
    normalized_content = normalize_reference_text(content)
    return normalized_law in normalized_content


def split_cross_law_query(query: str) -> tuple[str, str]:
    """Split "A법 안에 B법" style queries into source-law and target-law text."""
    match = CROSS_LAW_QUERY_SPLIT_PATTERN.search(query)
    if not match:
        return query, query
    return match.group(1), match.group(2)


def collect_referenced_laws(chunks: list[Document]) -> list[str]:
    """Collect law names referenced inside the corpus."""
    laws: list[str] = []
    seen: set[str] = set()
    for doc in chunks:
        for law, _, _ in extract_legal_references(doc.page_content[:6000]):
            normalized = normalize_reference_text(law)
            if normalized in seen:
                continue
            seen.add(normalized)
            laws.append(law)
    return laws


def retrieve_cross_law_mentions(
    chunks: list[Document],
    query: str,
    candidate_docs: list[Document],
    per_pair_limit: int = 2,
) -> list[Document]:
    """Find source-law chunks that mention another law named in the query.

    This handles questions like "A법 안에 B법이 언급된 조항이 있나?" before
    following the referenced B-law article itself.
    """
    if not query_has_cross_law_mention_intent(query):
        return []

    source_query, target_query = split_cross_law_query(query)
    source_laws = {
        source_law_name(str(doc.metadata.get("source", "")))
        for doc in chunks
        if doc.metadata.get("source")
    }
    source_laws = {
        law for law in source_laws if law and law_name_matches_query(source_query, law)
    }
    if not source_laws:
        source_laws = {
            law
            for law in {
                source_law_name(str(doc.metadata.get("source", "")))
                for doc in chunks
                if doc.metadata.get("source")
            }
            if law and law_name_matches_query(query, law)
        }
    referenced_laws = {
        law for law in collect_referenced_laws(chunks) if law_name_matches_query(target_query, law)
    }
    if not referenced_laws:
        referenced_laws = {
            law for law in collect_referenced_laws(chunks) if law_name_matches_query(query, law)
        }
    if not source_laws or not referenced_laws:
        return []

    max_hybrid_score = max(
        [float(doc.metadata.get("hybrid_score") or doc.metadata.get("score") or 0.0) for doc in candidate_docs]
        or [1.0]
    )
    mention_docs: list[Document] = []
    added_keys: set[tuple[str, str, int]] = set()

    for source_law in sorted(source_laws):
        for target_law in sorted(referenced_laws):
            if normalize_reference_text(source_law) == normalize_reference_text(target_law):
                continue
            matches = []
            for doc in chunks:
                doc_source_law = source_law_name(str(doc.metadata.get("source", "")))
                if normalize_reference_text(doc_source_law) != normalize_reference_text(source_law):
                    continue
                if not law_name_matches_content(target_law, doc.page_content):
                    continue
                matches.append(
                    (
                        doc,
                        token_overlap_ratio(query, doc.page_content)
                        + token_overlap_ratio(target_law, doc.page_content),
                        len(doc.page_content),
                    )
                )

            def rank_match(item: tuple[Document, float, int]) -> tuple[float, int]:
                doc, mention_score, content_length = item
                article_title_score = (
                    0.3
                    if re.search(r"제\s*\d+\s*조(?:의\s*\d+)?\s*[\(（]", doc.page_content)
                    else 0.0
                )
                return mention_score + article_title_score, -content_length

            for doc, mention_score, _ in sorted(matches, key=rank_match, reverse=True)[
                :per_pair_limit
            ]:
                key = doc_key(doc)
                if key in added_keys:
                    continue
                added_keys.add(key)
                mention_doc = clone_document(doc)
                mention_doc.metadata = {
                    **mention_doc.metadata,
                    "score": round(max_hybrid_score, 4),
                    "hybrid_score": round(max_hybrid_score, 4),
                    "mention_score": round(mention_score, 4),
                    "retrieval": "legal_mention",
                    "source_law": source_law,
                    "mentioned_law": target_law,
                }
                mention_docs.append(mention_doc)

    return mention_docs


def collect_legal_references(
    query: str,
    candidate_docs: list[Document],
) -> list[tuple[str, str, str]]:
    """Collect legal references without letting weak candidates expand context."""
    references: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for reference in extract_legal_references(query):
        if reference not in seen:
            seen.add(reference)
            references.append(reference)

    for doc in candidate_docs[:6]:
        if not candidate_supports_reference_expansion(query, doc):
            continue
        for reference in extract_legal_references(doc.page_content[:4000]):
            law, _, paragraph = reference
            if not paragraph and not law_name_is_query_related(query, law):
                continue
            if reference in seen:
                continue
            seen.add(reference)
            references.append(reference)

    return references


def legal_reference_match_score(
    doc: Document,
    law: str,
    article: str,
    paragraph: str = "",
) -> float:
    """Score whether a chunk contains the target law and article."""
    source_text = str(doc.metadata.get("source", ""))
    content = doc.page_content
    normalized_law = normalize_reference_text(law)
    normalized_source = normalize_reference_text(source_text)
    normalized_content = normalize_reference_text(content)

    law_in_source = normalized_law in normalized_source
    law_in_content = normalized_law in normalized_content
    if not (law_in_source or law_in_content):
        return 0.0

    if not re.search(rf"제\s*{re.escape(article)}\s*조", content):
        return 0.0

    score = 1.0
    if law_in_content:
        score += 0.3
    if re.search(rf"제\s*{re.escape(article)}\s*조\s*[\(（]", content):
        score += 0.4
    if paragraph and (
        re.search(rf"제\s*{re.escape(paragraph)}\s*항", content)
        or "①" in content
    ):
        score += 0.2
    score += min(token_overlap_ratio(law, content), 0.2)
    return score


def retrieve_legal_references(
    chunks: list[Document],
    query: str,
    candidate_docs: list[Document],
    per_reference_limit: int = 1,
) -> list[Document]:
    """Follow legal references found in query/candidates to matching chunks."""
    references = collect_legal_references(query, candidate_docs)
    if not references:
        return []
    specifically_referenced_laws = {
        normalize_reference_text(law)
        for law, _, paragraph in references
        if paragraph
    }
    references = [
        reference
        for reference in references
        if reference[2] or normalize_reference_text(reference[0]) not in specifically_referenced_laws
    ]

    existing_keys = {doc_key(doc) for doc in candidate_docs}
    max_hybrid_score = max(
        [float(doc.metadata.get("hybrid_score") or doc.metadata.get("score") or 0.0) for doc in candidate_docs]
        or [1.0]
    )
    reference_docs: list[Document] = []
    added_keys: set[tuple[str, str, int]] = set()

    for law, article, paragraph in references:
        matches = [
            (doc, legal_reference_match_score(doc, law, article, paragraph))
            for doc in chunks
        ]
        ranked_matches = sorted(
            [(doc, score) for doc, score in matches if score > 0],
            key=lambda item: (
                item[1],
                token_overlap_ratio(query, item[0].page_content),
            ),
            reverse=True,
        )
        for doc, reference_score in ranked_matches[:per_reference_limit]:
            key = doc_key(doc)
            if key in existing_keys or key in added_keys:
                continue
            added_keys.add(key)
            reference_doc = clone_document(doc)
            reference_doc.metadata = {
                **reference_doc.metadata,
                "score": round(max_hybrid_score, 4),
                "hybrid_score": round(max_hybrid_score, 4),
                "reference_score": round(reference_score, 4),
                "retrieval": "legal_reference",
                "referenced_law": law,
                "referenced_article": article,
                "referenced_paragraph": paragraph,
            }
            reference_docs.append(reference_doc)

    return reference_docs


def reference_doc_used_in_answer(doc: Document, answer: str) -> bool:
    """Return True when a followed legal reference is actually used in the answer."""
    normalized_answer = normalize_reference_text(answer)
    referenced_law = str(doc.metadata.get("referenced_law", ""))
    if referenced_law and normalize_reference_text(referenced_law) in normalized_answer:
        return True

    article_titles = re.findall(r"제\s*\d+\s*조\s*[\(（]([^）)]+)[\)）]", doc.page_content)
    for title in article_titles:
        title_tokens = [
            token
            for token in TOKEN_PATTERN.findall(title)
            if len(token) >= 3
        ]
        if any(normalize_reference_text(token) in normalized_answer for token in title_tokens):
            return True

    return False
