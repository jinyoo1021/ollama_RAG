"""Chainlit UI for the local PDF RAG chatbot."""

from __future__ import annotations

import asyncio
import os

import chainlit as cl
from chainlit.config import FILES_DIRECTORY

from src.chat_manager import ChatManager, stream_without_inline_sources
from src.llm_client import stream_chat
from src.rag_service import (
    RagResources,
    close_rag_resources,
    format_turn_sources,
    load_rag_resources,
    normalize_final_answer,
    prepare_turn,
    save_uploaded_pdfs,
)

SESSION_CHAT = "chat_manager"
SESSION_RESOURCES = "rag_resources"
UPLOAD_COMMANDS = {"/upload", "upload"}


def _ensure_files_directory() -> None:
    """Ensure Chainlit's local upload/element storage parent exists."""
    FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)


_ensure_files_directory()


def _should_rebuild_index() -> bool:
    return os.getenv("CHAINLIT_REBUILD_INDEX", "false").lower() == "true"


async def _load_resources() -> RagResources:
    return await asyncio.to_thread(
        load_rag_resources,
        rebuild=_should_rebuild_index(),
        build_if_missing=True,
    )


async def _set_resources(resources: RagResources, reset_history: bool = False) -> None:
    cl.user_session.set(SESSION_RESOURCES, resources)
    if reset_history or cl.user_session.get(SESSION_CHAT) is None:
        cl.user_session.set(SESSION_CHAT, ChatManager())


def _pdf_attachments(message: cl.Message) -> list[object]:
    files: list[object] = []
    for element in message.elements or []:
        mime = str(getattr(element, "mime", "") or "")
        path = str(getattr(element, "path", "") or "")
        if mime == "application/pdf" or path.lower().endswith(".pdf"):
            files.append(element)
    return files


async def _rebuild_after_upload(files: list[object]) -> RagResources:
    saved_paths = await asyncio.to_thread(save_uploaded_pdfs, files)
    status = cl.Message(
        content=(
            f"PDF {len(saved_paths)}개를 저장했습니다. "
            "새 인덱스를 만드는 중입니다..."
        )
    )
    await status.send()
    current_resources = cl.user_session.get(SESSION_RESOURCES)
    cl.user_session.set(SESSION_RESOURCES, None)
    await asyncio.to_thread(close_rag_resources, current_resources)
    resources = await asyncio.to_thread(
        load_rag_resources,
        rebuild=True,
        build_if_missing=True,
    )
    await _set_resources(resources, reset_history=True)
    status.content = (
        f"업로드 완료: PDF {len(resources.pdf_files)}개, "
        f"청크 {len(resources.chunks)}개로 인덱스를 갱신했습니다."
    )
    await status.update()
    return resources


async def _ask_for_pdf_upload() -> None:
    _ensure_files_directory()
    files = await cl.AskFileMessage(
        content="업로드할 PDF를 선택해주세요.",
        accept=["application/pdf"],
        max_size_mb=100,
        max_files=10,
        timeout=300,
    ).send()
    if not files:
        await cl.Message(content="업로드가 취소되었습니다.").send()
        return
    try:
        await _rebuild_after_upload(list(files))
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI for local setup.
        await cl.Message(content=f"PDF 업로드 처리 중 오류가 발생했습니다.\n\n`{exc}`").send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """Load retrieval resources once per browser chat session."""
    try:
        resources = await _load_resources()
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI for local setup.
        await cl.Message(
            content=(
                "RAG 리소스를 불러오지 못했습니다.\n\n"
                f"- 오류: `{exc}`\n"
                "- `data/pdfs/`에 PDF가 있는지 확인한 뒤 다시 실행해주세요."
            )
        ).send()
        return

    await _set_resources(resources, reset_history=True)

    pdf_count = len(resources.pdf_files)
    chunk_count = len(resources.chunks)
    await cl.Message(
        content=(
            f"PDF {pdf_count}개, 청크 {chunk_count}개를 불러왔습니다.\n\n"
            "질문을 입력하면 문서 근거와 출처를 함께 답변합니다.\n\n"
            "PDF를 추가하려면 `/upload`를 입력하세요."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer one user message with streaming output and metadata sources."""
    resources = cl.user_session.get(SESSION_RESOURCES)
    chat_manager = cl.user_session.get(SESSION_CHAT)
    if resources is None or chat_manager is None:
        await cl.Message(
            content="세션이 아직 준비되지 않았습니다. 새로고침 후 다시 질문해주세요."
        ).send()
        return

    query = message.content.strip()
    attached_pdfs = _pdf_attachments(message)
    if attached_pdfs:
        try:
            resources = await _rebuild_after_upload(attached_pdfs)
            chat_manager = cl.user_session.get(SESSION_CHAT)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI for local setup.
            await cl.Message(content=f"PDF 업로드 처리 중 오류가 발생했습니다.\n\n`{exc}`").send()
            return

    if query.lower() in UPLOAD_COMMANDS:
        await _ask_for_pdf_upload()
        return

    if not query:
        return

    try:
        turn = await asyncio.to_thread(prepare_turn, resources, query, chat_manager)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI for local setup.
        await cl.Message(content=f"검색 중 오류가 발생했습니다.\n\n`{exc}`").send()
        return

    if not turn.docs:
        await cl.Message(content="문서에서 해당 정보를 찾을 수 없습니다.").send()
        return

    answer_message = cl.Message(content="")
    await answer_message.send()

    answer = ""
    try:
        for text in stream_without_inline_sources(stream_chat(turn.messages)):
            answer += text
            await answer_message.stream_token(text)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI for local setup.
        answer_message.content = f"답변 생성 중 오류가 발생했습니다.\n\n`{exc}`"
        await answer_message.update()
        return

    final_answer = normalize_final_answer(answer)
    chat_manager.add_turn(query, final_answer)
    sources = format_turn_sources(resources, turn, final_answer)
    answer_message.content = f"{final_answer}\n\n{sources}"
    await answer_message.update()
