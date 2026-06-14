"""RAG (semantic search) request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentType


class IndexResponse(BaseModel):
    document_id: str
    chunk_count: int
    status: str = "indexed"


class RemoveResponse(BaseModel):
    document_id: str
    removed_points: int
    status: str = "removed"


class RagSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    # Optional metadata filters applied at vector-search time
    company_name: str | None = Field(default=None, max_length=255)
    document_type: DocumentType | None = None
    document_id: str | None = Field(default=None, max_length=36)


class RagHit(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float
    title: str
    company_name: str
    document_type: DocumentType


class RagSearchResponse(BaseModel):
    query: str
    retrieved: int
    reranked: int
    results: list[RagHit]


class RagContextResponse(BaseModel):
    document_id: str
    chunk_count: int
    chunks: list[RagHit]
