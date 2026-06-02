"""Ollama client wrapper for Korean answers."""

from collections.abc import Iterable

import ollama

from config import OLLAMA_MODEL


def chat(messages: list[dict[str, str]], model: str = OLLAMA_MODEL) -> str:
    """Call Ollama and return the full response text."""
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def stream_chat(
    messages: list[dict[str, str]],
    model: str = OLLAMA_MODEL,
) -> Iterable[str]:
    """Call Ollama and yield streamed response fragments."""
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        text = chunk.get("message", {}).get("content", "")
        if text:
            yield text
