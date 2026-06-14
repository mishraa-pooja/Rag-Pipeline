"""Document request/response schemas.

Mass-assignment defense: every response model uses an explicit field allow-list,
and request models use `extra="forbid"`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentType


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    company_name: str
    document_type: DocumentType
    uploaded_by: str
    original_filename: str
    content_type: str
    size_bytes: int
    is_indexed: bool
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentResponse]


class DocumentSearchQuery(BaseModel):
    """Used by GET /documents/search via query params (FastAPI will unpack)."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=255)
    company_name: str | None = Field(default=None, max_length=255)
    document_type: DocumentType | None = None
    uploaded_by: str | None = Field(default=None, max_length=36)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=10_000)
