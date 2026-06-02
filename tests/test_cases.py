"""Local tests for the Phase 2 pipeline helpers."""

from types import SimpleNamespace

from langchain_core.documents import Document

import config
import src.rag_service as rag_service_module
import src.retrieval.hybrid as retriever_module
from src.chat_manager import ChatManager, strip_inline_sources, stream_without_inline_sources
from src.indexer import make_chunk_ids, split_markdown_documents
from src.pdf_loader import clean_metadata, extract_printed_page_number
from src.rag_service import (
    PreparedTurn,
    RagResources,
    build_source_references,
    close_rag_resources,
    normalize_final_answer,
)
from src.retrieval.document_utils import normalize_scores
from src.retrieval.legal_references import (
    candidate_supports_reference_expansion,
    extract_legal_references,
    law_name_is_query_related,
    reference_doc_used_in_answer,
    retrieve_cross_law_mentions,
    retrieve_legal_references,
)
from src.retrieval.hybrid import expand_query, merge_hybrid_results
from src.retrieval.debug import RetrievalDebugTrace, format_retrieval_debug
from src.retrieval.source_resolver import (
    extract_named_terms,
    format_source_pages,
    get_source_pages,
    is_no_evidence_answer,
)


def test_project_config_uses_explicit_ollama_tag() -> None:
    assert config.OLLAMA_MODEL == "exaone3.5:7.8b"


def test_save_uploaded_pdfs_copies_files_without_overwriting(tmp_path, monkeypatch) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    source_pdf = upload_dir / "incoming.pdf"
    source_pdf.write_bytes(b"%PDF-1.7")
    data_dir = tmp_path / "pdfs"
    monkeypatch.setattr(rag_service_module, "DATA_DIR", data_dir)

    first = rag_service_module.save_uploaded_pdfs(
        [SimpleNamespace(name="../테스트.pdf", path=str(source_pdf), type="application/pdf")]
    )
    second = rag_service_module.save_uploaded_pdfs(
        [SimpleNamespace(name="../테스트.pdf", path=str(source_pdf), type="application/pdf")]
    )

    assert first[0] == data_dir / "테스트.pdf"
    assert second[0] == data_dir / "테스트_2.pdf"
    assert first[0].read_bytes() == b"%PDF-1.7"


def test_build_source_references_returns_ui_ready_snippets() -> None:
    resources = RagResources(
        vectorstore=None,
        chunks=[],
        pdf_files=[],
        show_source_names=True,
    )
    turn = PreparedTurn(
        retrieval_query="퇴직급여 우선변제",
        docs=[
            Document(
                page_content="퇴직급여등은 조세ㆍ공과금 및 다른 채권에 우선하여 변제되어야 한다.",
                metadata={
                    "source": "/tmp/근로자퇴직급여 보장법.pdf",
                    "page": 4,
                    "score": 0.9,
                    "hybrid_score": 0.9,
                },
            )
        ],
        messages=[],
    )

    refs = build_source_references(resources, turn, "퇴직급여등은 다른 채권에 우선하여 변제됩니다.")

    assert len(refs) == 1
    assert refs[0].source_name == "근로자퇴직급여 보장법"
    assert refs[0].page == "4"
    assert "우선하여 변제" in refs[0].preview


def test_close_rag_resources_closes_vectorstore_client() -> None:
    client = SimpleNamespace(closed=False)

    def close() -> None:
        client.closed = True

    client.close = close
    resources = RagResources(
        vectorstore=SimpleNamespace(_client=client),
        chunks=[],
        pdf_files=[],
        show_source_names=False,
    )

    close_rag_resources(resources)

    assert client.closed


def test_clean_metadata_keeps_only_allowed_scalar_values() -> None:
    metadata = clean_metadata(
        {
            "source": "sample.pdf",
            "page": 3,
            "h1": "개요",
            "ignored": "drop",
            "title": ["list", "value"],
            "none": None,
        }
    )

    assert metadata == {
        "source": "sample.pdf",
        "page": 3,
        "h1": "개요",
        "title": "['list', 'value']",
    }


def test_extract_printed_page_number_prefers_visible_page_label() -> None:
    assert extract_printed_page_number("22\n본문입니다.") == 22
    assert extract_printed_page_number("본문입니다.\n22") == 22


def test_split_markdown_documents_preserves_page_metadata() -> None:
    docs = [
        Document(
            page_content="# 제목\n\n본문입니다.\n\n## 세부\n\n내용입니다.",
            metadata={"source": "sample.pdf", "page": 1},
        )
    ]

    chunks = split_markdown_documents(docs)

    assert chunks
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)
    assert any(chunk.metadata.get("h1") == "제목" for chunk in chunks)


def test_split_markdown_documents_keeps_page_context_for_numbered_lists() -> None:
    docs = [
        Document(
            page_content=(
                "1. 탈중앙성\n설명\n2. 투명성\n설명\n"
                "3. 불변성\n설명\n4. 가용성\n설명"
            ),
            metadata={"source": "sample.pdf", "page": 22},
        )
    ]

    chunks = split_markdown_documents(docs)

    assert any(
        "1. 탈중앙성" in chunk.page_content and "4. 가용성" in chunk.page_content
        for chunk in chunks
    )


def test_make_chunk_ids_is_stable_for_same_input() -> None:
    chunks = [Document(page_content="hello", metadata={"source": "a.pdf", "page": 1})]

    assert make_chunk_ids(chunks) == make_chunk_ids(chunks)


def test_source_pages_come_from_metadata_not_content() -> None:
    docs = [
        Document(page_content="제12조 적용 범위", metadata={"page": 2, "score": 0.5}),
        Document(page_content="다른 내용", metadata={"page": 2, "score": 0.48}),
        Document(page_content="제13조", metadata={"page": 3, "score": 0.44}),
    ]

    assert get_source_pages(docs) == ["2"]
    assert format_source_pages(docs) == "출처: p.2"


def test_source_pages_prefer_keyword_evidence_over_raw_vector_score() -> None:
    docs = [
        Document(
            page_content="근로기준법 목적과 정의",
            metadata={"page": 1, "score": 0.8},
        ),
        Document(
            page_content=(
                "제11조 적용 범위. 동거하는 친족만을 사용하는 사업 또는 "
                "사업장과 가사 사용인에 대하여는 적용하지 아니한다."
            ),
            metadata={"page": 2, "score": 0.7},
        ),
    ]

    assert format_source_pages(docs, "근로기준법의 적용범위? 그 예외는?") == "출처: p.2"


def test_source_pages_include_all_evidence_pages() -> None:
    docs = [
        Document(
            page_content="제1조 목적. 근로조건의 기준과 근로자 보호.",
            metadata={"page": 1, "score": 0.7},
        ),
        Document(
            page_content="제11조 적용 범위. 상시 5명 이상의 근로자를 사용하는 사업장.",
            metadata={"page": 2, "score": 0.69},
        ),
    ]

    answer = "근로조건의 기준과 근로자 보호 목적, 그리고 상시 5명 이상 사업장 적용 범위입니다."

    assert (
        format_source_pages(docs, "근로기준법 적용 범위와 목적은?", answer)
        == "출처: p.1, p.2"
    )


def test_source_pages_can_show_single_source_name_for_multi_pdf_corpus() -> None:
    docs = [
        Document(
            page_content="제43조의2 체불사업주 명단 공개",
            metadata={"source": "/tmp/근로기준법(법률)(제20520호)_long.pdf", "page": 6, "score": 0.9},
        )
    ]

    assert (
        format_source_pages(docs, "체불사업주 명단공개", show_source_names=True)
        == "출처: 근로기준법 p.6"
    )


def test_source_pages_follow_answer_text_for_follow_up_exception() -> None:
    docs = [
        Document(
            page_content="근로기준법 목적과 정의",
            metadata={"page": 1, "score": 0.8},
        ),
        Document(
            page_content="제11조 적용 범위. 동거하는 친족만을 사용하는 사업장과 가사 사용인은 제외된다.",
            metadata={"page": 2, "score": 0.72},
        ),
    ]
    answer = "예외는 동거하는 친족만을 사용하는 사업장과 가사 사용인입니다."

    assert format_source_pages(docs, "근로기준법 적용범위\n그 예외는?", answer) == "출처: p.2"


def test_source_pages_include_named_terms_used_in_answer_without_hard_coding() -> None:
    docs = [
        Document(
            page_content="알파플랫폼과 1인 미디어의 유행",
            metadata={"page": 120, "score": 0.49},
        ),
        Document(
            page_content="동영상 플랫폼인 베타TV는 수익 모델을 가지고 있다.",
            metadata={"page": 121, "score": 0.59},
        ),
        Document(
            page_content="알파플랫폼에도 저작권을 무시한 콘텐츠가 다수 게시되고 있다.",
            metadata={"page": 130, "score": 0.45},
        ),
    ]
    answer = "명시적으로 언급된 예시로는 알파플랫폼과 베타TV가 있습니다."

    assert format_source_pages(docs, "문화 콘텐츠 기관", answer) == "출처: p.120, p.121"


def test_extract_named_terms_finds_generic_korean_and_latin_names() -> None:
    terms = extract_named_terms("명시적으로 언급된 예시로는 알파플랫폼과 BetaTV가 있습니다.")

    assert "알파플랫폼" in terms
    assert "betatv" in terms


def test_expand_query_does_not_use_domain_specific_terms() -> None:
    query = "문화 콘텐츠에서 현재 주로 사용하는 기관은 어딘지"

    assert expand_query(query) == query


def test_retrieve_hybrid_expands_query_once(monkeypatch) -> None:
    class FakeVectorstore:
        query = ""

        def similarity_search_with_relevance_scores(self, query: str, k: int):
            self.query = query
            return []

    calls: list[str] = []

    def fake_expand_query(query: str) -> str:
        calls.append(query)
        return f"expanded:{query}"

    vectorstore = FakeVectorstore()
    monkeypatch.setattr(retriever_module, "expand_query", fake_expand_query)

    retriever_module.retrieve_hybrid(vectorstore, [], "원문 질문")

    assert calls == ["원문 질문"]
    assert vectorstore.query == "expanded:원문 질문"


def test_retrieve_hybrid_populates_debug_trace(monkeypatch) -> None:
    vector_doc = Document(
        page_content="벡터 후보 본문",
        metadata={"source": "/tmp/샘플.pdf", "page": 1, "score": 0.8},
    )
    bm25_doc = Document(
        page_content="BM25 후보 본문",
        metadata={"source": "/tmp/샘플.pdf", "page": 2, "score": 0.7, "bm25_score": 1.0},
    )

    class FakeVectorstore:
        def similarity_search_with_relevance_scores(self, query: str, k: int):
            return [(vector_doc, 0.8)]

    monkeypatch.setattr(retriever_module, "expand_query", lambda query: f"확장 {query}")
    monkeypatch.setattr(
        retriever_module,
        "retrieve_bm25",
        lambda chunks, query, top_k: [bm25_doc],
    )
    monkeypatch.setattr(
        retriever_module,
        "retrieve_legal_references",
        lambda chunks, query, candidates: [],
    )

    trace = RetrievalDebugTrace()
    docs = retriever_module.retrieve_hybrid(
        FakeVectorstore(),
        [],
        "원문 질문",
        debug_trace=trace,
    )

    assert docs
    assert trace.original_query == "원문 질문"
    assert trace.retrieval_query == "확장 원문 질문"
    assert trace.stages["vector"]
    assert trace.stages["bm25"]
    assert trace.stages["merged"]
    assert trace.stages["final"]


def test_format_retrieval_debug_shows_source_page_and_scores() -> None:
    trace = RetrievalDebugTrace(
        original_query="질문",
        retrieval_query="질문",
        top_k=10,
        rerank_top_k=3,
        use_reranker=False,
    )
    trace.record(
        "final",
        [
            Document(
                page_content="디버그 출력에 표시될 본문입니다.",
                metadata={
                    "source": "/tmp/근로기준법(법률)(제20520호)_long.pdf",
                    "page": 6,
                    "score": 0.9,
                    "hybrid_score": 0.8,
                    "retrieval": "hybrid",
                },
            )
        ],
    )

    output = format_retrieval_debug(trace)

    assert "[debug] retrieval" in output
    assert "근로기준법 p.6" in output
    assert "hybrid=0.8000" in output


def test_extract_legal_references_finds_quoted_law_article() -> None:
    text = "「근로자퇴직급여 보장법」 제12조제1항에 따른 퇴직급여등"

    assert extract_legal_references(text) == [("근로자퇴직급여 보장법", "12", "1")]


def test_retrieve_legal_references_follows_cross_document_article() -> None:
    labor_doc = Document(
        page_content=(
            "제43조의2 체불사업주 명단 공개. "
            "「근로자퇴직급여 보장법」 제12조제1항에 따른 퇴직급여등을 지급하지 아니한 사업주."
        ),
        metadata={"source": "/tmp/근로기준법.pdf", "page": 6, "score": 0.9, "hybrid_score": 0.9},
    )
    retirement_doc = Document(
        page_content=(
            "근로자퇴직급여 보장법 제12조(퇴직급여등의 우선변제) "
            "① 퇴직급여등은 조세ㆍ공과금 및 다른 채권에 우선하여 변제되어야 한다."
        ),
        metadata={"source": "/tmp/근로자퇴직급여 보장법.pdf", "page": 4},
    )

    docs = retrieve_legal_references(
        [labor_doc, retirement_doc],
        "체불사업주 명단공개와 퇴직급여",
        [labor_doc],
    )

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "/tmp/근로자퇴직급여 보장법.pdf"
    assert docs[0].metadata["retrieval"] == "legal_reference"


def test_retrieve_cross_law_mentions_finds_source_law_clause() -> None:
    labor_doc = Document(
        page_content=(
            "근로기준법 제43조의2(체불사업주 명단 공개) "
            "「근로자퇴직급여 보장법」 제12조제1항에 따른 퇴직급여등을 "
            "지급하지 아니한 사업주를 체불사업주라 한다."
        ),
        metadata={"source": "/tmp/근로기준법.pdf", "page": 6},
    )
    retirement_doc = Document(
        page_content=(
            "근로자퇴직급여 보장법 제12조(퇴직급여등의 우선변제) "
            "① 퇴직급여등은 다른 채권에 우선하여 변제되어야 한다."
        ),
        metadata={"source": "/tmp/근로자퇴직급여 보장법.pdf", "page": 4},
    )

    docs = retrieve_cross_law_mentions(
        [labor_doc, retirement_doc],
        "근로기준법 안에 근로퇴직자 급여 보장법이 포함되어있는 법령이 존재해?",
        [],
    )

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "/tmp/근로기준법.pdf"
    assert docs[0].metadata["page"] == 6
    assert docs[0].metadata["retrieval"] == "legal_mention"


def test_retrieve_cross_law_mentions_can_boost_existing_candidate() -> None:
    labor_doc = Document(
        page_content=(
            "근로기준법 제43조의2(체불사업주 명단 공개) "
            "「근로자퇴직급여 보장법」 제12조제1항에 따른 퇴직급여등."
        ),
        metadata={"source": "/tmp/근로기준법.pdf", "page": 6, "score": 0.3},
    )
    retirement_doc = Document(
        page_content="근로자퇴직급여 보장법 제12조(퇴직급여등의 우선변제)",
        metadata={"source": "/tmp/근로자퇴직급여 보장법.pdf", "page": 4},
    )

    docs = retrieve_cross_law_mentions(
        [labor_doc, retirement_doc],
        "근로기준법 안에 근로자퇴직급여 보장법이 포함되어있는 법령이 존재해?",
        [labor_doc],
    )

    assert len(docs) == 1
    assert docs[0].metadata["retrieval"] == "legal_mention"
    assert docs[0].metadata["hybrid_score"] == 0.3


def test_legal_mention_anchor_follows_referenced_article_even_with_spaced_query() -> None:
    labor_doc = Document(
        page_content=(
            "근로기준법 제43조의2(체불사업주 명단 공개) "
            "「근로자퇴직급여 보장법」 제12조제1항에 따른 퇴직급여등."
        ),
        metadata={"source": "/tmp/근로기준법.pdf", "page": 6, "retrieval": "legal_mention"},
    )
    retirement_doc = Document(
        page_content=(
            "근로자퇴직급여 보장법 제12조(퇴직급여등의 우선변제) "
            "① 퇴직급여등은 조세ㆍ공과금 및 다른 채권에 우선하여 변제되어야 한다."
        ),
        metadata={"source": "/tmp/근로자퇴직급여 보장법.pdf", "page": 4},
    )

    docs = retrieve_legal_references(
        [labor_doc, retirement_doc],
        "근로기준법 안에 근로퇴직자 급여 보장법이 포함되어있는 법령이 존재해?",
        [labor_doc],
    )

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "/tmp/근로자퇴직급여 보장법.pdf"
    assert docs[0].metadata["referenced_article"] == "12"


def test_legal_reference_ignores_weak_unrelated_candidate() -> None:
    weak_candidate = Document(
        page_content=(
            "근로기준법 출석 의무와 다른 임금 지급의무. "
            "「근로자퇴직급여 보장법」 제2조제5호에 따른 급여."
        ),
        metadata={"source": "/tmp/근로기준법.pdf", "page": 6, "score": 0.4},
    )
    retirement_doc = Document(
        page_content="근로자퇴직급여 보장법 제2조(정의) 퇴직급여제도란 급여를 지급하는 제도이다.",
        metadata={"source": "/tmp/근로자퇴직급여 보장법.pdf", "page": 1},
    )

    assert candidate_supports_reference_expansion("근로기준법 출석의 의무", weak_candidate)
    assert not law_name_is_query_related("근로기준법 출석의 의무", "근로자퇴직급여 보장법")
    assert (
        retrieve_legal_references(
            [weak_candidate, retirement_doc],
            "근로기준법 출석의 의무",
            [weak_candidate],
        )
        == []
    )


def test_format_source_pages_groups_multiple_pdf_sources() -> None:
    docs = [
        Document(
            page_content="체불사업주 명단 공개와 퇴직급여등",
            metadata={"source": "/tmp/근로기준법(법률)(제20520호)_long.pdf", "page": 6, "score": 0.9},
        ),
        Document(
            page_content="제12조(퇴직급여등의 우선변제) 다른 채권에 우선하여 변제된다.",
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 4,
                "score": 0.88,
                "retrieval": "legal_reference",
                "referenced_law": "근로자퇴직급여 보장법",
            },
        ),
    ]
    answer = "체불사업주 명단 공개에는 퇴직급여등이 포함되고, 퇴직급여등은 우선변제됩니다."

    assert (
        format_source_pages(docs, "체불사업주 명단공개와 퇴직급여", answer)
        == "출처: 근로기준법 p.6; 근로자퇴직급여 보장법 p.4"
    )


def test_legal_reference_source_is_omitted_when_answer_does_not_use_it() -> None:
    docs = [
        Document(
            page_content="제43조의2 체불사업주 명단 공개",
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호)_long.pdf",
                "page": 6,
                "score": 0.9,
            },
        ),
        Document(
            page_content="제12조(퇴직급여등의 우선변제) 퇴직급여등은 다른 채권에 우선하여 변제된다.",
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 4,
                "score": 0.88,
                "retrieval": "legal_reference",
                "referenced_law": "근로자퇴직급여 보장법",
            },
        ),
    ]
    answer = "체불사업주 명단공개는 임금 등을 체불한 사업주의 인적사항을 공개하는 제도입니다."

    assert not reference_doc_used_in_answer(docs[1], answer)
    assert (
        format_source_pages(
            docs,
            "체불사업주 명단공개는 무엇인가?",
            answer,
            show_source_names=True,
        )
        == "출처: 근로기준법 p.6"
    )


def test_blockchain_no_evidence_answer_does_not_cite_law_pdfs() -> None:
    docs = [
        Document(
            page_content="근로기준법 제13조 보고 출석의 의무",
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호)_long.pdf",
                "page": 2,
                "score": 0.9,
                "hybrid_score": 0.9,
            },
        ),
        Document(
            page_content="근로자퇴직급여 보장법 제12조 퇴직급여등의 우선변제",
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 4,
                "score": 0.8,
                "hybrid_score": 0.8,
            },
        ),
    ]
    answer = "문서 내용에서 블록체인 유형별 특징에 대한 관련 내용을 확인할 수 없습니다."

    assert is_no_evidence_answer(answer)
    assert (
        format_source_pages(
            docs,
            "블록체인 유형별 특징",
            answer,
            show_source_names=True,
        )
        == "출처: 관련 문서 없음"
    )


def test_single_answer_source_does_not_cite_other_pdf_in_multi_pdf_results() -> None:
    docs = [
        Document(
            page_content="근로기준법 제13조 보고 출석의 의무. 보고하거나 출석하여야 한다.",
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호)_long.pdf",
                "page": 2,
                "score": 0.9,
                "hybrid_score": 0.9,
            },
        ),
        Document(
            page_content="근로자퇴직급여 보장법 제12조 퇴직급여등의 우선변제.",
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 4,
                "score": 0.72,
                "hybrid_score": 0.72,
            },
        ),
    ]
    answer = (
        "근로기준법 제13조에 따르면 사용자 또는 근로자는 요구가 있으면 "
        "필요한 사항을 보고하거나 출석해야 합니다."
    )

    assert (
        format_source_pages(
            docs,
            "근로기준법 출석의 의무",
            answer,
            show_source_names=True,
        )
        == "출처: 근로기준법 p.2"
    )


def test_definition_answer_uses_answer_law_article_to_avoid_other_law_sources() -> None:
    docs = [
        Document(
            page_content=(
                "근로기준법 제2조(정의) ① 이 법에서 사용하는 용어의 뜻은 다음과 같다. "
                "1. 근로자란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 "
                "근로를 제공하는 사람을 말한다."
            ),
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호).pdf",
                "page": 1,
                "score": 0.95,
                "hybrid_score": 0.95,
            },
        ),
        Document(
            page_content=(
                "근로자퇴직급여 보장법 제2조(정의) 근로자와 사용자의 퇴직급여제도에 "
                "관한 용어를 정의한다."
            ),
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 1,
                "score": 0.92,
                "hybrid_score": 0.92,
            },
        ),
        Document(
            page_content="근로자퇴직급여 보장법 적용 범위와 퇴직급여 설정.",
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 2,
                "score": 0.85,
                "hybrid_score": 0.85,
            },
        ),
    ]
    answer = (
        "근로자란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 "
        "근로를 제공하는 사람을 말합니다. 이 정의는 「근로기준법」 "
        "제2조제1항제1호에 따라 정해집니다."
    )

    assert (
        format_source_pages(
            docs,
            "근로자란?",
            answer,
            show_source_names=True,
        )
        == "출처: 근로기준법 p.1"
    )


def test_working_condition_standard_uses_article_pages_not_generic_term_pages() -> None:
    docs = [
        Document(
            page_content=(
                "근로기준법 제3조(근로조건의 기준) 이 법에서 정하는 근로조건은 "
                "최저기준이다. 제4조(근로조건의 결정) 근로조건은 근로자와 사용자가 "
                "동등한 지위에서 자유의사에 따라 결정하여야 한다."
            ),
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호).pdf",
                "page": 1,
                "score": 0.9,
                "hybrid_score": 0.9,
            },
        ),
        Document(
            page_content="근로기준법 제17조 근로조건의 명시. 임금, 소정근로시간, 휴일.",
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호).pdf",
                "page": 3,
                "score": 0.88,
                "hybrid_score": 0.88,
            },
        ),
    ]
    answer = (
        "근로조건의 기준은 「근로기준법」 제3조에서 정하며, 이 기준은 최저 수준입니다. "
        "또한 제4조에 따라 근로자와 사용자는 동등한 지위에서 자유롭게 "
        "근로조건을 결정해야 합니다."
    )

    assert (
        format_source_pages(
            docs,
            "근로조건의 기준",
            answer,
            show_source_names=True,
        )
        == "출처: 근로기준법 p.1"
    )


def test_no_answer_suppresses_sources_even_when_retrieval_has_multiple_pdfs() -> None:
    docs = [
        Document(
            page_content="근로기준법 제1조 목적",
            metadata={
                "source": "/tmp/근로기준법(법률)(제20520호)_long.pdf",
                "page": 1,
                "score": 0.95,
            },
        ),
        Document(
            page_content="근로자퇴직급여 보장법 제1조 목적",
            metadata={
                "source": "/tmp/근로자퇴직급여 보장법(법률)(제21135호).pdf",
                "page": 1,
                "score": 0.93,
            },
        ),
    ]
    answer = "문서에서 해당 질문에 대한 답변을 제공할 수 없습니다."

    assert is_no_evidence_answer(answer)
    assert (
        format_source_pages(
            docs,
            "블록체인 유형별 특징",
            answer,
            show_source_names=True,
        )
        == "출처: 관련 문서 없음"
    )


def test_no_evidence_answer_suppresses_unrelated_sources() -> None:
    docs = [
        Document(
            page_content="근로기준법 목적과 정의",
            metadata={"source": "/tmp/근로기준법.pdf", "page": 1, "score": 0.8},
        )
    ]
    answer = (
        "문서 내용에서 블록체인 유형별 특징에 대한 정보는 명시적으로 언급되어 있지 않습니다. "
        "따라서 이 질문에 대한 답변을 제공할 수 없습니다."
    )

    assert is_no_evidence_answer(answer)
    assert format_source_pages(docs, "블록체인 유형별 특징", answer) == "출처: 관련 문서 없음"


def test_no_evidence_answer_handles_difficult_to_answer_phrase() -> None:
    docs = [
        Document(
            page_content="근로기준법 목적과 정의",
            metadata={"source": "/tmp/근로기준법.pdf", "page": 1, "score": 0.8},
        )
    ]
    answer = (
        "문서 내용에서 블록체인 유형별 특징에 대한 정보는 제공되지 않았습니다. "
        "따라서 해당 질문에 대해 답변 드리기 어렵습니다. 다른 정보나 문서가 필요합니다."
    )

    assert is_no_evidence_answer(answer)
    assert format_source_pages(docs, "블록체인 유형별 특징", answer) == "출처: 관련 문서 없음"


def test_no_evidence_answer_handles_general_knowledge_fallback_phrase() -> None:
    docs = [
        Document(
            page_content="근로기준법 제17조 근로조건의 명시",
            metadata={"source": "/tmp/근로기준법.pdf", "page": 3, "score": 0.8},
        )
    ]
    answer = (
        "문서 내용에서 블록체인의 특징에 대한 직접적인 정보는 제공되지 않았습니다. "
        "따라서 블록체인의 일반적인 특징을 기반으로 답변 드리겠습니다."
    )

    assert is_no_evidence_answer(answer)
    assert normalize_final_answer(answer) == "문서에서 해당 정보를 찾을 수 없습니다."
    assert (
        format_source_pages(docs, "블록체인의 4가지 특징", answer, show_source_names=True)
        == "출처: 관련 문서 없음"
    )


def test_no_evidence_answer_handles_not_included_phrase() -> None:
    docs = [
        Document(
            page_content="근로기준법 목적과 정의",
            metadata={"source": "/tmp/근로기준법.pdf", "page": 1, "score": 0.8},
        )
    ]
    answer = (
        "문서 내용에 블록체인 유형별 특징에 대한 정보는 포함되어 있지 않습니다. "
        "따라서 해당 질문에 대해 답변 드릴 수 없습니다."
    )

    assert is_no_evidence_answer(answer)
    assert format_source_pages(docs, "블록체인 유형별 특징", answer) == "출처: 관련 문서 없음"


def test_partial_answer_with_evidence_keeps_sources() -> None:
    docs = [
        Document(
            page_content="탈중앙성 투명성 불변성 가용성",
            metadata={"page": 22, "score": 1.0, "hybrid_score": 1.0},
        )
    ]
    answer = "문서에서 명시적으로 모두 나열한 부분은 찾을 수 없습니다. 하지만 탈중앙성, 투명성은 확인됩니다."

    assert not is_no_evidence_answer(answer)
    assert format_source_pages(docs, "블록체인 특징", answer) == "출처: p.22"


def test_normalize_scores_handles_equal_values() -> None:
    assert normalize_scores([2.0, 2.0]) == [1.0, 1.0]
    assert normalize_scores([1.0, 3.0]) == [0.0, 1.0]


def test_merge_hybrid_results_combines_vector_and_bm25_scores() -> None:
    vector_doc = Document(
        page_content="같은 청크",
        metadata={"source": "a.pdf", "page": 1, "score": 0.9},
    )
    bm25_doc = Document(
        page_content="같은 청크",
        metadata={"source": "a.pdf", "page": 1, "bm25_score": 0.8},
    )

    merged = merge_hybrid_results([vector_doc], [bm25_doc])

    assert len(merged) == 1
    assert merged[0].metadata["retrieval"] == "hybrid"
    assert merged[0].metadata["vector_score"] == 1.0
    assert merged[0].metadata["bm25_score"] == 0.8


def test_source_pages_use_hybrid_score_to_avoid_bm25_noise() -> None:
    docs = [
        Document(
            page_content="탈중앙성 투명성 불변성 가용성",
            metadata={"page": 22, "score": 1.0, "hybrid_score": 1.0},
        ),
        Document(
            page_content="탈중앙성 투명성 불변성 가용성 다른 맥락",
            metadata={"page": 88, "score": 0.9, "hybrid_score": 0.3},
        ),
    ]
    answer = "블록체인의 4가지 기술적 특징은 탈중앙성, 투명성, 불변성, 가용성입니다."

    assert format_source_pages(docs, "블록체인의 4가지 기술적 특징", answer) == "출처: p.22"


def test_strip_inline_sources_removes_model_citations() -> None:
    answer = "적용됩니다. (출처: 근로기준법 제12조) 추가 설명입니다. (문서 참조: 제12조)"

    assert strip_inline_sources(answer) == "적용됩니다. 추가 설명입니다."


def test_stream_without_inline_sources_handles_split_citations() -> None:
    chunks = ["답변입니다. (출", "처: 제12조) 계속됩니다. (p.", "12 참조)"]

    assert "".join(stream_without_inline_sources(chunks)) == "답변입니다. 계속됩니다."


def test_chat_manager_keeps_bounded_turn_history() -> None:
    manager = ChatManager(max_turns=2)
    manager.add_turn("q1", "a1")
    manager.add_turn("q2", "a2")
    manager.add_turn("q3", "a3")

    assert manager.messages() == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_chat_manager_build_messages_does_not_store_current_query() -> None:
    manager = ChatManager(max_turns=2)
    manager.add_turn("이전 질문", "이전 답변")

    messages = manager.build_messages("문서 내용", "현재 질문", "출처: p.2")

    assert manager.messages() == [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"},
    ]
    assert messages[-1]["content"].endswith("[질문]\n현재 질문")
    assert sum("현재 질문" in message["content"] for message in messages) == 1


def test_chat_manager_builds_contextual_retrieval_query() -> None:
    manager = ChatManager(max_turns=3)
    manager.add_turn("적용 지역은?", "답변")

    assert manager.build_retrieval_query("그 예외는?") == "적용 지역은?\n그 예외는?"
