"""Shared RAG service helpers for CLI and Chainlit UI."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any

from langchain_core.documents import Document

from config import DATA_DIR, REBUILD_INDEX
from src.chat_manager import ChatManager, build_rag_messages
from src.indexer import build_index, create_or_load_vectorstore, load_chunks
from src.pdf_loader import find_pdf_files
from src.retrieval.hybrid import format_context, retrieve_hybrid
from src.retrieval.source_resolver import (
    format_source_name,
    format_source_pages,
    is_no_evidence_answer,
    select_source_docs,
)


@dataclass
class RagResources:
    """Loaded retrieval resources shared across chat turns."""

    vectorstore: Any
    chunks: list[Document]
    pdf_files: list[Path]
    show_source_names: bool


@dataclass
class PreparedTurn:
    """Retrieved documents and model messages for one user question."""

    retrieval_query: str
    docs: list[Document]
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class SourceReference:
    """Compact source evidence for UI display."""

    source_name: str
    source_path: str
    page: str
    preview: str


NO_EVIDENCE_RESPONSE = "문서에서 해당 정보를 찾을 수 없습니다."


def collect_pdfs() -> list[Path]:
    """Return PDF files from the configured corpus directory."""
    pdf_files = find_pdf_files(DATA_DIR)
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {DATA_DIR}")
    return pdf_files


def has_multiple_sources(chunks: list[Document], pdf_files: list[Path] | None = None) -> bool:
    """Return True when the active corpus appears to contain multiple PDFs."""
    indexed_sources = {
        str(chunk.metadata.get("source", ""))
        for chunk in chunks
        if getattr(chunk, "metadata", None) and chunk.metadata.get("source")
    }
    if pdf_files is None:
        try:
            pdf_files = collect_pdfs()
        except RuntimeError:
            pdf_files = []
    pdf_sources = {str(path) for path in pdf_files}
    return len(indexed_sources | pdf_sources) > 1


def load_rag_resources(
    rebuild: bool = REBUILD_INDEX,
    build_if_missing: bool = False,
) -> RagResources:
    """Load or build the vectorstore, chunk cache, and source display settings."""
    pdf_files = collect_pdfs()
    if rebuild:
        vectorstore, chunks = build_index(pdf_files, rebuild=True)
    else:
        try:
            vectorstore = create_or_load_vectorstore(rebuild=False)
            chunks = load_chunks()
        except FileNotFoundError:
            if not build_if_missing:
                raise
            vectorstore, chunks = build_index(pdf_files, rebuild=True)

    return RagResources(
        vectorstore=vectorstore,
        chunks=chunks,
        pdf_files=pdf_files,
        show_source_names=has_multiple_sources(chunks, pdf_files),
    )


def prepare_turn(
    resources: RagResources,
    query: str,
    chat_manager: ChatManager | None = None,
    debug_trace: Any | None = None,
) -> PreparedTurn:
    """Retrieve context and build model messages for one turn."""
    retrieval_query = (
        chat_manager.build_retrieval_query(query) if chat_manager is not None else query
    )
    docs = retrieve_hybrid(
        resources.vectorstore,
        resources.chunks,
        retrieval_query,
        debug_trace=debug_trace,
    )
    if not docs:
        return PreparedTurn(retrieval_query=retrieval_query, docs=[], messages=[])

    preliminary_sources = format_source_pages(
        docs,
        retrieval_query,
        show_source_names=resources.show_source_names,
    )
    context = format_context(docs)
    messages = (
        chat_manager.build_messages(context, query, preliminary_sources)
        if chat_manager is not None
        else build_rag_messages(context, query, preliminary_sources)
    )
    return PreparedTurn(
        retrieval_query=retrieval_query,
        docs=docs,
        messages=messages,
    )


def format_turn_sources(
    resources: RagResources,
    turn: PreparedTurn,
    answer: str = "",
) -> str:
    """Format source pages for a completed answer."""
    return format_source_pages(
        turn.docs,
        turn.retrieval_query,
        answer,
        show_source_names=resources.show_source_names,
    )


def normalize_final_answer(answer: str) -> str:
    """Collapse unsupported model fallback text into the canonical no-evidence answer."""
    normalized = answer.strip()
    if is_no_evidence_answer(normalized):
        return NO_EVIDENCE_RESPONSE
    return normalized


def close_rag_resources(resources: RagResources | None) -> None:
    """Release Chroma resources before rebuilding the persistent database."""
    if resources is None:
        return

    client = getattr(resources.vectorstore, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()

    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    gc.collect()


def _safe_pdf_name(name: str) -> str:
    """Return a safe local PDF filename while preserving readable Korean names."""
    filename = Path(name).name.strip() or "uploaded.pdf"
    filename = re.sub(r"[\x00-\x1f]", "", filename)
    filename = filename.replace("/", "_").replace("\\", "_")
    if Path(filename).suffix.lower() != ".pdf":
        filename = f"{Path(filename).stem}.pdf"
    return filename


def _unique_pdf_path(filename: str) -> Path:
    """Avoid overwriting existing PDFs with the same uploaded filename."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    index = 2
    while True:
        candidate = DATA_DIR / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _validate_pdf_file(path: Path, display_name: str) -> None:
    """Validate that the upload has a PDF file signature."""
    with path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise ValueError(f"PDF 형식이 아닌 파일입니다: {display_name}")


def save_uploaded_pdfs(files: list[Any]) -> list[Path]:
    """Copy Chainlit-uploaded PDF files into data/pdfs."""
    saved_paths: list[Path] = []
    for file in files:
        source_path = Path(str(getattr(file, "path", "")))
        file_type = str(getattr(file, "type", "") or getattr(file, "mime", ""))
        file_name = str(getattr(file, "name", source_path.name))
        if source_path.suffix.lower() != ".pdf" and file_type != "application/pdf":
            raise ValueError(f"PDF 파일만 업로드할 수 있습니다: {file_name}")
        if not source_path.exists():
            raise FileNotFoundError(f"업로드 파일을 찾을 수 없습니다: {file_name}")

        _validate_pdf_file(source_path, file_name)
        target = _unique_pdf_path(_safe_pdf_name(file_name))
        shutil.copyfile(source_path, target)
        saved_paths.append(target)
    return saved_paths


def _preview_text(text: str, limit: int = 420) -> str:
    """Normalize a chunk preview for source display."""
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= limit:
        return preview
    return f"{preview[: limit - 3].rstrip()}..."


def build_source_references(
    resources: RagResources,
    turn: PreparedTurn,
    answer: str = "",
    max_refs: int = 6,
) -> list[SourceReference]:
    """Return selected evidence snippets for Chainlit source UI elements."""
    if answer and is_no_evidence_answer(answer):
        return []

    refs: list[SourceReference] = []
    seen: set[tuple[str, str]] = set()
    selected_docs = select_source_docs(turn.docs, turn.retrieval_query, answer)
    for doc in selected_docs:
        source_path = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", "?"))
        if not source_path or page == "?" or (source_path, page) in seen:
            continue
        seen.add((source_path, page))
        refs.append(
            SourceReference(
                source_name=format_source_name(source_path),
                source_path=source_path,
                page=page,
                preview=_preview_text(doc.page_content),
            )
        )
        if len(refs) >= max_refs:
            break
    return refs
