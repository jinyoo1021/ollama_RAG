"""Shared retrieval evaluation cases and metrics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document

from src.retrieval.source_resolver import format_source_name

TermMatchMode = Literal["all", "any"]


@dataclass(frozen=True)
class ExpectedRef:
    """Expected source/page evidence for one eval case."""

    source: str | None = None
    pages: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalCase:
    """One retrieval/source evaluation case."""

    id: str
    query: str
    expected_refs: tuple[ExpectedRef, ...] = ()
    expected_terms: tuple[str, ...] = ()
    term_match: TermMatchMode = "all"
    tags: tuple[str, ...] = field(default_factory=tuple)


LEGAL_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="labor_attendance_duty",
        query="근로기준법 출석의 의무",
        expected_refs=(ExpectedRef("근로기준법", ("2",)),),
        expected_terms=("출석", "보고", "제13조"),
        tags=("legal", "source"),
    ),
    EvalCase(
        id="overdue_employer_disclosure",
        query="체불사업주 명단공개는 무엇인가?",
        expected_refs=(ExpectedRef("근로기준법", ("6", "7")),),
        expected_terms=("체불사업주", "명단 공개", "제43조의2"),
        tags=("legal", "source"),
    ),
    EvalCase(
        id="overdue_employer_and_retirement_priority",
        query="체불사업주 명단공개는 무엇이고, 퇴직급여는 어떻게 되는가?",
        expected_refs=(
            ExpectedRef("근로기준법", ("6",)),
            ExpectedRef("근로자퇴직급여 보장법", ("4",)),
        ),
        expected_terms=("체불사업주", "퇴직급여등", "우선하여 변제"),
        tags=("legal", "cross-law", "source"),
    ),
    EvalCase(
        id="cross_law_mention",
        query="근로기준법 안에 근로자퇴직급여 보장법이 포함되어있는 법령이 존재해?",
        expected_refs=(
            ExpectedRef("근로기준법", ("6",)),
            ExpectedRef("근로자퇴직급여 보장법", ("4",)),
        ),
        expected_terms=("제43조의2", "제12조", "퇴직급여등"),
        tags=("legal", "cross-law", "source"),
    ),
    EvalCase(
        id="retirement_priority",
        query="퇴직급여등의 우선변제는 무엇인가?",
        expected_refs=(ExpectedRef("근로자퇴직급여 보장법", ("4",)),),
        expected_terms=("퇴직급여등", "조세", "다른 채권", "우선"),
        tags=("legal", "source"),
    ),
    EvalCase(
        id="labor_standard_scope_exception",
        query="근로기준법의 적용범위와 예외는?",
        expected_refs=(ExpectedRef("근로기준법", ("1", "2")),),
        expected_terms=("상시 5명", "동거하는 친족", "가사 사용인"),
        tags=("legal", "source"),
    ),
)


BLOCKCHAIN_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="blockchain_technical_features",
        query="블록체인의 4가지 기술적 특징",
        expected_refs=(ExpectedRef(None, ("22",)),),
        expected_terms=("탈중앙성", "투명성", "불변성", "가용성"),
        tags=("blockchain", "source"),
    ),
    EvalCase(
        id="culture_content_platforms",
        query="문화 콘텐츠에서 현재 주로 사용하는 기관은 어딘지",
        expected_refs=(ExpectedRef(None, ("120", "121")),),
        expected_terms=("유튜브", "아프리카TV", "포털", "플랫폼"),
        term_match="any",
        tags=("blockchain", "source"),
    ),
    EvalCase(
        id="direct_creator_consumer",
        query="창작자와 소비자 직접 거래로 어떤 효과가 기대되는지",
        expected_refs=(ExpectedRef(None, ("120",)),),
        expected_terms=("직접 거래", "수익", "창작자"),
        term_match="any",
        tags=("blockchain", "source"),
    ),
    EvalCase(
        id="steemit_example",
        query="블록체인 기반 블로그 플랫폼의 대표적인 예는?",
        expected_refs=(ExpectedRef(None, ("123",)),),
        expected_terms=("스팀잇", "Steemit", "보상"),
        term_match="any",
        tags=("blockchain", "source"),
    ),
    EvalCase(
        id="transparency_impact",
        query="개인끼리 믿고 거래할 수 있는 힘은 무엇에서 나왔나",
        expected_refs=(ExpectedRef(None, ("51",)),),
        expected_terms=("투명성", "거래 장부", "공개"),
        term_match="any",
        tags=("blockchain", "source"),
    ),
)


DEFAULT_EVAL_CASES: tuple[EvalCase, ...] = LEGAL_EVAL_CASES + BLOCKCHAIN_EVAL_CASES


def get_eval_cases(case_set: str = "all") -> list[EvalCase]:
    """Return eval cases by group name."""
    if case_set == "all":
        return list(DEFAULT_EVAL_CASES)
    if case_set == "legal":
        return list(LEGAL_EVAL_CASES)
    if case_set == "blockchain":
        return list(BLOCKCHAIN_EVAL_CASES)
    raise ValueError(f"Unknown case set: {case_set}")


def source_page_pairs(docs: list[Document]) -> set[tuple[str, str]]:
    """Return normalized source/page pairs from retrieved docs."""
    pairs: set[tuple[str, str]] = set()
    for doc in docs:
        source = format_source_name(str(doc.metadata.get("source", "")))
        page = str(doc.metadata.get("page", "?"))
        pairs.add((source, page))
    return pairs


def expected_ref_hit(docs: list[Document], expected_refs: tuple[ExpectedRef, ...]) -> bool:
    """Return True when all expected source/page refs are present."""
    pairs = source_page_pairs(docs)
    for expected in expected_refs:
        for page in expected.pages:
            if expected.source is None:
                if not any(actual_page == page for _, actual_page in pairs):
                    return False
                continue
            if not any(
                expected.source in actual_source and actual_page == page
                for actual_source, actual_page in pairs
            ):
                return False
    return True


def any_expected_ref_hit(
    docs: list[Document],
    expected_refs: tuple[ExpectedRef, ...],
) -> bool:
    """Return True when at least one expected source/page ref is present."""
    if not expected_refs:
        return True
    pairs = source_page_pairs(docs)
    for expected in expected_refs:
        for page in expected.pages:
            if expected.source is None:
                if any(actual_page == page for _, actual_page in pairs):
                    return True
                continue
            if any(
                expected.source in actual_source and actual_page == page
                for actual_source, actual_page in pairs
            ):
                return True
    return False


def unexpected_refs(
    docs: list[Document],
    expected_refs: tuple[ExpectedRef, ...],
) -> list[str]:
    """Return refs that fall outside the expected source/page set."""
    if not expected_refs:
        return []

    allowed_pages_by_source: dict[str | None, set[str]] = {}
    for expected in expected_refs:
        allowed_pages_by_source.setdefault(expected.source, set()).update(expected.pages)

    extras: list[str] = []
    for source, page in sorted(source_page_pairs(docs)):
        if None in allowed_pages_by_source:
            if page not in allowed_pages_by_source[None]:
                extras.append(f"{source} p.{page}")
            continue
        if not any(
            expected_source in source and page in pages
            for expected_source, pages in allowed_pages_by_source.items()
            if expected_source is not None
        ):
            extras.append(f"{source} p.{page}")
    return extras


def term_hit(text: str, terms: tuple[str, ...], mode: TermMatchMode = "all") -> bool:
    """Return True when expected terms are present in text."""
    if not terms:
        return True
    lowered = text.lower()
    checks = [term.lower() in lowered for term in terms]
    return all(checks) if mode == "all" else any(checks)


def evaluate_retrieval_docs(docs: list[Document], case: EvalCase) -> dict[str, object]:
    """Evaluate retrieved docs against an eval case."""
    context = "\n".join(doc.page_content for doc in docs)
    ref_hit = expected_ref_hit(docs, case.expected_refs)
    keyword_hit = term_hit(context, case.expected_terms, case.term_match)
    pairs = sorted(f"{source} p.{page}" for source, page in source_page_pairs(docs))
    return {
        "id": case.id,
        "query": case.query,
        "ref_hit": ref_hit,
        "term_hit": keyword_hit,
        "both_hit": ref_hit and keyword_hit,
        "top_refs": [
            f"{format_source_name(str(doc.metadata.get('source', '')))} p.{doc.metadata.get('page', '?')}"
            for doc in docs[:3]
        ],
        "final_refs": pairs,
    }


def latency_stats(samples: list[float]) -> dict[str, float]:
    """Return common latency stats for elapsed-second samples."""
    if not samples:
        return {
            "count": 0.0,
            "avg": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": float(len(samples)),
        "avg": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }
