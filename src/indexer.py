"""Index building utilities for chunking, embedding, and Chroma storage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHROMA_COLLECTION_NAME,
    CHUNK_STORE_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DEVICE,
    EMBEDDING_MODEL,
    MARKDOWN_HEADERS,
    MIN_CHUNK_CHARS,
    PAGE_CONTEXT_MAX_CHARS,
    REBUILD_INDEX,
    VECTOR_DB_DIR,
)
from src.pdf_loader import clean_metadata, load_pdf_with_pages


def create_embedding_function() -> HuggingFaceEmbeddings:
    """Create normalized local embeddings for Korean document retrieval."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_markdown_documents(docs: Iterable[Document]) -> list[Document]:
    """Split Markdown by headings first, then split oversized sections."""
    return split_markdown_documents_with_config(
        docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def split_markdown_documents_with_config(
    docs: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split Markdown documents with explicit chunk parameters."""
    original_docs = list(docs)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    section_docs: list[Document] = []
    for doc in original_docs:
        split_docs = header_splitter.split_text(doc.page_content)
        for split_doc in split_docs:
            split_doc.metadata = clean_metadata({**doc.metadata, **split_doc.metadata})
            section_docs.append(split_doc)

    chunks = recursive_splitter.split_documents(section_docs)
    for chunk in chunks:
        chunk.metadata = clean_metadata(chunk.metadata)

    meaningful_chunks = [
        chunk for chunk in chunks if len(chunk.page_content.strip()) >= MIN_CHUNK_CHARS
    ]
    page_context_docs = [
        Document(page_content=doc.page_content, metadata=clean_metadata(doc.metadata))
        for doc in original_docs
        if MIN_CHUNK_CHARS <= len(doc.page_content.strip()) <= PAGE_CONTEXT_MAX_CHARS
    ]
    return [*page_context_docs, *(meaningful_chunks or chunks)]


def make_chunk_ids(chunks: Iterable[Document]) -> list[str]:
    """Build stable-ish IDs from source, page, chunk order, and content."""
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        raw = "|".join(
            [
                str(chunk.metadata.get("source", "")),
                str(chunk.metadata.get("page", "")),
                str(index),
                chunk.page_content,
            ]
        )
        ids.append(hashlib.sha1(raw.encode("utf-8")).hexdigest())
    return ids


def load_pdf_documents(pdf_paths: Iterable[str | Path]) -> list[Document]:
    """Load all PDFs into page-level Markdown documents."""
    docs: list[Document] = []
    for path in pdf_paths:
        docs.extend(load_pdf_with_pages(path))
    return docs


def build_chunks(pdf_paths: Iterable[str | Path]) -> list[Document]:
    """Load and split PDFs into Chroma-ready chunks."""
    docs = load_pdf_documents(pdf_paths)
    if not docs:
        raise ValueError("No readable PDF text was found.")
    return split_markdown_documents(docs)


def save_chunks(chunks: list[Document]) -> None:
    """Persist chunks so BM25 can be rebuilt in --reuse-index mode."""
    CHUNK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "page_content": chunk.page_content,
            "metadata": clean_metadata(chunk.metadata),
        }
        for chunk in chunks
    ]
    with CHUNK_STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def load_chunks() -> list[Document]:
    """Load persisted chunks for lexical retrieval."""
    if not CHUNK_STORE_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNK_STORE_PATH} does not exist. Run indexing once before reuse."
        )
    with CHUNK_STORE_PATH.open(encoding="utf-8") as file:
        payload = json.load(file)
    return [
        Document(
            page_content=str(item.get("page_content", "")),
            metadata=clean_metadata(dict(item.get("metadata", {}))),
        )
        for item in payload
        if isinstance(item, dict)
    ]


def create_or_load_vectorstore(
    chunks: list[Document] | None = None,
    rebuild: bool = REBUILD_INDEX,
) -> Chroma:
    """Create a Chroma vectorstore from chunks or load an existing one."""
    embeddings = create_embedding_function()

    if rebuild:
        if chunks is None:
            raise ValueError("chunks are required when rebuild=True.")
        if VECTOR_DB_DIR.exists():
            shutil.rmtree(VECTOR_DB_DIR)
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        (VECTOR_DB_DIR / ".gitkeep").touch()
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=make_chunk_ids(chunks),
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=CHROMA_COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"{VECTOR_DB_DIR} does not exist. Run indexing once before reuse."
        )

    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(
    pdf_paths: Iterable[str | Path],
    rebuild: bool = REBUILD_INDEX,
) -> tuple[Chroma, list[Document]]:
    """Build or reload the vector index for the supplied PDFs."""
    chunks = build_chunks(pdf_paths) if rebuild else None
    if chunks is not None:
        save_chunks(chunks)
    vectorstore = create_or_load_vectorstore(chunks=chunks, rebuild=rebuild)
    return vectorstore, chunks or load_chunks()
