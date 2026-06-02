"""PDF loading utilities for the Phase 2 RAG pipeline."""

from pathlib import Path
from typing import Any

import pymupdf4llm
from langchain_core.documents import Document


ALLOWED_METADATA_KEYS = {"source", "page", "title", "h1", "h2", "h3"}
SIMPLE_METADATA_TYPES = (str, int, float, bool)


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only Chroma-safe scalar metadata used by the RAG pipeline."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key not in ALLOWED_METADATA_KEYS or value is None:
            continue
        if isinstance(value, SIMPLE_METADATA_TYPES):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def extract_printed_page_number(text: str) -> int | None:
    """Extract the page number printed in the PDF body when it is obvious."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = []
    if lines:
        candidates.extend([lines[0], lines[-1]])

    for candidate in candidates:
        if candidate.isdigit() and len(candidate) <= 4:
            return int(candidate)
    return None


def load_pdf_with_pages(file_path: str | Path) -> list[Document]:
    """Convert a PDF to page-level Markdown documents with source metadata."""
    path = Path(file_path)
    page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    docs: list[Document] = []
    for index, chunk in enumerate(page_chunks, start=1):
        raw_metadata = dict(chunk.get("metadata", {}))
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        page = (
            extract_printed_page_number(text)
            or raw_metadata.get("page")
            or raw_metadata.get("page_number")
            or index
        )

        metadata = clean_metadata(
            {
                "source": str(path),
                "page": page,
                "title": raw_metadata.get("title"),
            }
        )
        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def find_pdf_files(directory: str | Path) -> list[Path]:
    """Return PDF files in deterministic order."""
    return sorted(Path(directory).glob("*.pdf"))
