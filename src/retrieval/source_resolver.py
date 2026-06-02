"""Source page selection and answer attribution helpers."""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

from langchain_core.documents import Document

from config import SOURCE_MAX_PAGES, SOURCE_SCORE_MARGIN
from src.retrieval.document_utils import (
    LAW_ARTICLE_PATTERN,
    merge_unique_documents,
    token_overlap_ratio,
)
from src.retrieval.legal_references import reference_doc_used_in_answer


NO_EVIDENCE_PATTERNS = (
    re.compile(r"문서에서\s*해당\s*정보를\s*찾을\s*수\s*없습니다"),
    re.compile(r"정보는\s*(?:명시적으로\s*)?(?:제공|언급)되어\s*있지\s*않습니다"),
    re.compile(r"정보는\s*(?:직접적으로\s*|직접적인\s*)?(?:제공|언급)되지\s*않았습니다"),
    re.compile(r"정보는?\s*(?:포함|수록)되어\s*있지\s*않습니다"),
    re.compile(r"정보를\s*(?:확인|찾을)\s*수\s*없습니다"),
    re.compile(r"관련\s*(?:내용|정보)(?:을|를|가)?\s*(?:확인|찾)(?:할|을)?\s*수\s*없습니다"),
    re.compile(r"답변을\s*제공할\s*수\s*없습니다"),
    re.compile(r"답변\s*드릴\s*수\s*없습니다"),
    re.compile(r"답변\s*드리기\s*어렵습니다"),
    re.compile(r"답변하기\s*어렵습니다"),
    re.compile(r"질문에\s*대한\s*답변을\s*제공할\s*수\s*없습니다"),
    re.compile(r"일반적인\s*(?:내용|정보|지식|특징)을?\s*기반으로\s*답변"),
    re.compile(r"일반\s*지식(?:을|에)?\s*기반으로"),
)

IMPORTANT_TERMS = {
    "적용": 0.2,
    "범위": 0.2,
    "예외": 0.25,
    "제외": 0.2,
    "사업": 0.1,
    "사업장": 0.1,
    "가사": 0.15,
    "친족": 0.15,
    "국가": 0.1,
    "지역": 0.1,
}

KOREAN_NOUNISH_PATTERN = re.compile(r"[가-힣]+(?:TV|tv)?")
LATIN_ENTITY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9.+_-]{1,}")
KOREAN_PARTICLE_SUFFIXES = (
    "으로는",
    "적으로",
    "로는",
    "의",
    "과",
    "와",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
)
GENERIC_NAMED_TERMS = {
    "근로자",
    "사용자",
    "사업",
    "사업장",
    "직업",
    "종류",
    "관계",
    "임금",
    "목적",
    "근로",
    "근로조건",
    "명시적",
    "명시적으로",
    "언급된",
    "예시",
    "그리고",
    "상시",
    "이상",
    "적용",
    "범위",
    "보호",
    "제공",
    "사람",
    "기준",
    "조건",
    "법률",
    "조항",
}
QUOTED_LAW_NAME_PATTERN = re.compile(r"[「『]([^」』]{2,}?법)[」』]")
ARTICLE_NUMBER_PATTERN = re.compile(r"제\s*(\d+)\s*조")


def strip_korean_particle(term: str) -> str:
    """Remove a simple Korean particle suffix from a candidate term."""
    for suffix in KOREAN_PARTICLE_SUFFIXES:
        if term.endswith(suffix) and len(term) > len(suffix) + 1:
            return term[: -len(suffix)]
    return term


def extract_content_terms(text: str) -> set[str]:
    """Extract comparable terms from document or answer text."""
    terms = {term.lower() for term in LATIN_ENTITY_PATTERN.findall(text)}
    for term in KOREAN_NOUNISH_PATTERN.findall(text):
        stripped = strip_korean_particle(term)
        if len(stripped) >= 2:
            terms.add(stripped.lower())
    return terms


def is_no_evidence_answer(answer: str) -> bool:
    """Return True when the generated answer says the documents do not support it."""
    normalized = re.sub(r"\s+", " ", answer.strip())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in NO_EVIDENCE_PATTERNS)


def format_source_name(source: str) -> str:
    """Return a readable source name for multi-PDF citations."""
    if not source:
        return "문서"
    stem = unicodedata.normalize("NFC", Path(source).stem)
    stem = re.sub(r"\([^)]*\)", "", stem)
    stem = re.sub(r"_long$", "", stem)
    return stem.strip() or "문서"


def extract_named_terms(text: str) -> set[str]:
    """Extract answer terms that are likely to be concrete names or titles."""
    terms = {term.lower() for term in LATIN_ENTITY_PATTERN.findall(text)}
    for term in KOREAN_NOUNISH_PATTERN.findall(text):
        term = strip_korean_particle(term)
        if (
            len(term) >= 3
            and term not in GENERIC_NAMED_TERMS
            and not term.endswith(("니다", "습니다", "입니다"))
        ):
            terms.add(term.lower())
    return terms


def named_term_in_content(term: str, content: str) -> bool:
    """Match named terms by comparable tokens, not loose substrings."""
    return term.lower() in extract_content_terms(content)


def normalize_law_name(text: str) -> str:
    """Normalize spacing and wrapper punctuation in a Korean law name."""
    text = re.sub(r"[「」『』()\[\]]", "", text)
    return re.sub(r"\s+", "", text).strip().lower()


def source_or_content_has_law(doc: Document, law: str) -> bool:
    """Return True when a document appears to belong to or quote the law."""
    normalized_law = normalize_law_name(law)
    if not normalized_law:
        return False
    source = format_source_name(str(doc.metadata.get("source", "")))
    haystack = normalize_law_name(f"{source} {doc.page_content[:300]}")
    return normalized_law in haystack


def extract_law_names(text: str) -> set[str]:
    """Extract law names explicitly mentioned in an answer or query."""
    laws: set[str] = set()
    for match in QUOTED_LAW_NAME_PATTERN.finditer(text):
        law = match.group(1).strip()
        law = re.sub(r"\s+", " ", law)
        if len(normalize_law_name(law)) >= 3:
            laws.add(law)
    return laws


def extract_mentioned_source_laws(docs: list[Document], text: str) -> set[str]:
    """Return source law names from the corpus that are explicitly in text."""
    normalized_text = normalize_law_name(text)
    laws = extract_law_names(text)
    for doc in docs:
        source_law = format_source_name(str(doc.metadata.get("source", "")))
        normalized_source = normalize_law_name(source_law)
        if normalized_source and normalized_source in normalized_text:
            laws.add(source_law)
    return laws


def extract_article_numbers(text: str) -> set[str]:
    """Extract article numbers mentioned in generated answers."""
    return {match.group(1) for match in ARTICLE_NUMBER_PATTERN.finditer(text)}


def doc_contains_article(doc: Document, article: str) -> bool:
    """Return True when the doc text contains the requested article number."""
    return re.search(rf"제\s*{re.escape(article)}\s*조", doc.page_content) is not None


def score_evidence_doc(doc: Document, query: str = "") -> float:
    """Score a retrieved doc as a source candidate."""
    vector_score = float(doc.metadata.get("score") or 0.0)
    content = doc.page_content
    overlap_score = token_overlap_ratio(query, content)
    important_score = sum(
        weight
        for term, weight in IMPORTANT_TERMS.items()
        if term in query and term in content
    )
    article_score = 0.15 if LAW_ARTICLE_PATTERN.search(content) else 0.0

    return vector_score + overlap_score + important_score + article_score


def select_answer_source_docs(
    docs: list[Document],
    answer: str,
    query: str = "",
) -> list[Document]:
    """Pick docs whose text appears to support the generated answer."""
    scored_docs = [
        doc for doc in docs if isinstance(doc.metadata.get("score"), int | float)
    ]
    if not scored_docs:
        return docs[:1]

    answer_laws = extract_mentioned_source_laws(scored_docs, answer)
    article_numbers = extract_article_numbers(answer)
    article_selected: list[Document] = []
    if article_numbers:
        for doc in scored_docs:
            if not any(doc_contains_article(doc, article) for article in article_numbers):
                continue
            if answer_laws and not any(source_or_content_has_law(doc, law) for law in answer_laws):
                continue
            article_selected.append(doc)
        if article_selected:
            return merge_unique_documents(article_selected)

    overlap_scores = [
        (doc, token_overlap_ratio(answer, doc.page_content))
        for doc in scored_docs
    ]
    best_overlap = max(score for _, score in overlap_scores)
    if best_overlap <= 0:
        return []

    def retrieval_score(doc: Document) -> float:
        return float(doc.metadata.get("hybrid_score") or doc.metadata.get("score") or 0.0)

    best_vector_score = max(retrieval_score(doc) for doc, _ in overlap_scores)
    overlap_floor = max(0.08, best_overlap - 0.2)
    vector_floor = best_vector_score - 0.2
    query_floor = 0.15 if query else 0.0
    overlap_selected = [
        doc
        for doc, overlap_score in overlap_scores
        if (query_overlap := token_overlap_ratio(query, doc.page_content)) is not None
        if (
            (
                overlap_score >= overlap_floor
                or (query_overlap >= query_floor and overlap_score >= 0.08)
            )
            and retrieval_score(doc) >= vector_floor
            and (
                query_overlap >= query_floor
                or overlap_score >= best_overlap
            )
            and (
                not answer_laws
                or any(source_or_content_has_law(doc, law) for law in answer_laws)
            )
        )
    ]
    reference_selected = [
        doc
        for doc, _ in overlap_scores
        if (
            doc.metadata.get("retrieval") == "legal_reference"
            and reference_doc_used_in_answer(doc, answer)
        )
    ]
    if reference_selected:
        non_reference_support = [
            (doc, overlap_score)
            for doc, overlap_score in overlap_scores
            if doc.metadata.get("retrieval") != "legal_reference" and overlap_score > 0
        ]
        if non_reference_support:
            anchor_doc, _ = max(
                non_reference_support,
                key=lambda item: (item[1], retrieval_score(item[0])),
            )
            reference_selected = merge_unique_documents([anchor_doc], reference_selected)
    overlap_selected = merge_unique_documents(overlap_selected, reference_selected)

    named_selected: list[Document] = []
    for term in extract_named_terms(answer):
        matching_docs = [
            doc
            for doc in scored_docs
            if named_term_in_content(term, doc.page_content)
            and retrieval_score(doc) >= vector_floor
        ]
        if matching_docs:
            named_selected.append(
                max(matching_docs, key=lambda doc: float(doc.metadata["score"]))
            )

    if named_selected:
        named_pages = {doc.metadata.get("page") for doc in named_selected}
        return [
            *merge_unique_documents(named_selected, reference_selected),
            *[
                doc
                for doc in overlap_selected
                if doc.metadata.get("page") in named_pages
            ],
        ]

    return overlap_selected


def select_source_docs(
    docs: list[Document],
    query: str = "",
    answer: str = "",
) -> list[Document]:
    """Keep the strongest evidence docs for source display."""
    if answer:
        answer_docs = select_answer_source_docs(docs, answer, query)
        if answer_docs:
            return answer_docs

    scored_docs = [
        doc for doc in docs if isinstance(doc.metadata.get("score"), int | float)
    ]
    if not scored_docs:
        return docs[:1]

    ranked_docs = sorted(
        scored_docs,
        key=lambda doc: score_evidence_doc(doc, query),
        reverse=True,
    )
    best_score = score_evidence_doc(ranked_docs[0], query)
    threshold = best_score - SOURCE_SCORE_MARGIN
    selected = [
        doc
        for doc in ranked_docs
        if score_evidence_doc(doc, query) >= threshold
    ]
    return selected or ranked_docs[:1]


def page_sort_key(page: str) -> tuple[int, int | str]:
    """Sort numeric page labels naturally, then non-numeric labels."""
    return (0, int(page)) if page.isdigit() else (1, page)


def get_source_pages(
    docs: list[Document],
    query: str = "",
    answer: str = "",
) -> list[str]:
    """Return unique source PDF pages from the strongest retrieved docs."""
    pages: list[str] = []
    seen: set[str] = set()
    for doc in select_source_docs(docs, query, answer):
        page = str(doc.metadata.get("page", "?"))
        if page == "?" or page in seen:
            continue
        seen.add(page)
        pages.append(page)
        if SOURCE_MAX_PAGES is not None and len(pages) >= SOURCE_MAX_PAGES:
            break
    return sorted(pages, key=page_sort_key)


def format_source_pages(
    docs: list[Document],
    query: str = "",
    answer: str = "",
    show_source_names: bool = False,
) -> str:
    """Format source pages from metadata, not from the model answer."""
    if answer and is_no_evidence_answer(answer):
        return "출처: 관련 문서 없음"

    selected_docs = select_source_docs(docs, query, answer)
    if not selected_docs:
        return "출처: 페이지 정보 없음"

    sources = {
        str(doc.metadata.get("source", ""))
        for doc in selected_docs
        if doc.metadata.get("source")
    }
    if len(sources) <= 1:
        pages = []
        seen_pages: set[str] = set()
        for doc in selected_docs:
            page = str(doc.metadata.get("page", "?"))
            if page == "?" or page in seen_pages:
                continue
            seen_pages.add(page)
            pages.append(page)
            if SOURCE_MAX_PAGES is not None and len(pages) >= SOURCE_MAX_PAGES:
                break
        if not pages:
            return "출처: 페이지 정보 없음"
        if show_source_names and sources:
            source = next(iter(sources))
            return (
                f"출처: {format_source_name(source)} "
                + ", ".join(f"p.{page}" for page in sorted(pages, key=page_sort_key))
            )
        return "출처: " + ", ".join(f"p.{page}" for page in sorted(pages, key=page_sort_key))

    grouped_pages: dict[str, list[str]] = {}
    seen_refs: set[tuple[str, str]] = set()
    for doc in selected_docs:
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", "?"))
        if not source or page == "?" or (source, page) in seen_refs:
            continue
        seen_refs.add((source, page))
        grouped_pages.setdefault(source, []).append(page)
        if SOURCE_MAX_PAGES is not None and len(seen_refs) >= SOURCE_MAX_PAGES:
            break

    if not grouped_pages:
        return "출처: 페이지 정보 없음"

    groups = []
    for source in sorted(grouped_pages, key=format_source_name):
        pages = sorted(grouped_pages[source], key=page_sort_key)
        groups.append(f"{format_source_name(source)} " + ", ".join(f"p.{page}" for page in pages))
    return "출처: " + "; ".join(groups)
