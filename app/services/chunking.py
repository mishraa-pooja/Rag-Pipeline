"""Semantic chunking via LangChain's recursive character text splitter.

The splitter walks down a list of separators (paragraphs, sentences, words)
until each chunk fits below the target size, which produces chunks that
respect natural language boundaries — a good compromise between speed and
semantic coherence for finance text.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


@dataclass
class Chunk:
    index: int
    text: str


def chunk_text(text: str) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    raw_chunks = splitter.split_text(text or "")
    return [Chunk(index=i, text=c.strip()) for i, c in enumerate(raw_chunks) if c.strip()]
