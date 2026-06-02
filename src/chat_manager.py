"""Prompt helpers for document-grounded Korean answers."""

from collections import deque
from collections.abc import Iterable, Iterator
import re

from config import MAX_HISTORY_TURNS


SYSTEM_PROMPT = """당신은 문서 기반 한국어 질의응답 어시스턴트입니다.

규칙:
1. 제공된 [문서 내용]만 근거로 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고만 답하세요.
   문서에 없다고 판단한 뒤 일반 지식, 배경지식, 상식, 모델 지식으로 보충 설명하지 마세요.
3. 답변은 한국어로 명확하고 간결하게 작성하세요.
4. 답변 본문에 페이지 번호, 출처, 참조 표기를 직접 쓰지 마세요. 출처는 시스템이 별도로 표시합니다.
5. "제12조" 같은 조문 번호를 "p.12" 같은 페이지 번호로 바꾸지 마세요.
6. 사용자가 "기관"이라고 물었더라도 문서가 "플랫폼", "사이트", "서비스", "회사" 이름으로 설명하면 그 명칭을 답하세요.
7. 문서 내용에 구체적인 고유명사나 서비스명이 나오면 "명시적으로 언급된 예시"로 취급하세요.
8. 관련 고유명사가 문서 내용에 있으면 "명시적으로 언급되어 있지 않다"라고 답하지 마세요.
9. 사용자가 어디, 누구, 기관, 회사, 플랫폼, 서비스 등을 물으면 관련 고유명사를 우선적으로 찾아 답하세요.
10. 문서에 다른 법률 조항 참조와 그 참조 조항의 원문이 함께 제공되면 두 조항의 관계를 함께 설명하세요.
11. 법령의 "A를 제외하고는 B에 우선한다" 같은 우선순위 표현은 방향을 바꾸지 말고 원문 의미 그대로 설명하세요.
12. 사용자가 어떤 법 안에 다른 법이 "포함", "언급", "참조"되는지 물으면, 법률 자체가 편입되었다고 단정하지 말고 해당 조항에서 다른 법률을 참조한다고 설명하세요.
"""

INLINE_SOURCE_PATTERN = re.compile(
    r"\s*\((?:출처|참조|문서\s*참조|source|p\.\s*\d+|페이지)[^)]*\)",
    flags=re.IGNORECASE,
)
CROSS_LAW_MENTION_QUESTION_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9ㆍ·\s]+법.+(?:안에|내에|에서|중에).+[가-힣A-Za-z0-9ㆍ·\s]+법.+"
    r"(?:포함|들어\s*있|언급|참조|관련\s*조항|법령이?\s*존재|조항이?\s*존재)"
)


def build_rag_messages(
    context: str,
    query: str,
    source_pages: str = "",
    history: Iterable[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build a single-turn RAG prompt for Ollama chat models."""
    page_hint = ""
    if source_pages:
        page_hint = (
            f"\n\n[사용 가능한 실제 PDF 페이지]\n{source_pages}\n"
            "위 목록은 PDF metadata에서 가져온 실제 페이지입니다. "
            "답변 본문에는 페이지 번호를 쓰지 마세요."
        )
    question_hint = ""
    if CROSS_LAW_MENTION_QUESTION_PATTERN.search(query):
        question_hint = (
            "\n\n[질문 해석]\n"
            "사용자의 'A법 안에 B법이 포함/언급/참조되는가' 질문은 "
            "B법이 A법에 편입되었다는 뜻이 아니라, A법 조문 안에서 "
            "B법을 명시적으로 언급하거나 참조하는 조항이 있는지 묻는 뜻입니다. "
            "해당 조항이 있으면 '존재한다'고 답하고 조항명을 설명하세요.\n"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"[문서 내용]\n{context}{page_hint}{question_hint}\n\n[질문]\n{query}",
        },
    ]
    if history:
        messages[1:1] = list(history)
    return messages


def strip_inline_sources(answer: str) -> str:
    """Remove model-generated citation text; metadata sources are appended later."""
    return INLINE_SOURCE_PATTERN.sub("", answer).strip()


def stream_without_inline_sources(chunks: Iterable[str]) -> Iterator[str]:
    """Yield streamed text while suppressing model-generated parenthetical sources."""
    paren_buffer = ""
    buffering_paren = False
    pending_space = False

    for chunk in chunks:
        for char in chunk:
            if buffering_paren:
                paren_buffer += char
                if char == ")":
                    cleaned = strip_inline_sources(paren_buffer)
                    if cleaned:
                        yield cleaned
                    paren_buffer = ""
                    buffering_paren = False
                continue

            if char in "\r\n":
                pending_space = False
                yield char
                continue

            if char.isspace():
                pending_space = True
                continue

            if char == "(":
                paren_buffer = f"{' ' if pending_space else ''}{char}"
                buffering_paren = True
                pending_space = False
                continue

            if pending_space:
                yield " "
                pending_space = False
            yield char

    if paren_buffer:
        cleaned = strip_inline_sources(paren_buffer)
        if cleaned:
            yield cleaned


class ChatManager:
    """Bounded multi-turn chat history.

    max_turns is counted as user/assistant pairs, not raw messages.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        self.max_turns = max_turns
        self.history: deque[dict[str, str]] = deque(maxlen=max_turns * 2)

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})

    def messages(self) -> list[dict[str, str]]:
        return list(self.history)

    def build_messages(
        self,
        context: str,
        query: str,
        source_pages: str = "",
    ) -> list[dict[str, str]]:
        """Build messages without adding the current query to history."""
        return build_rag_messages(
            context=context,
            query=query,
            source_pages=source_pages,
            history=self.history,
        )

    def build_retrieval_query(self, query: str, previous_user_turns: int = 2) -> str:
        """Augment short follow-up questions with recent user context."""
        previous_questions = [
            message["content"]
            for message in self.history
            if message["role"] == "user"
        ][-previous_user_turns:]
        if not previous_questions:
            return query
        return "\n".join([*previous_questions, query])
