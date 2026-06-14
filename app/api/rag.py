"""RAG endpoints: index, remove, search, fetch document context.

Authorization rules:
- /rag/index-document            : `rag:index`  (analysts + admins)
- /rag/remove-document/{id}      : `rag:index`
- /rag/search                    : `rag:search` (clients are scoped to their company)
- /rag/context/{document_id}     : same view rules as GET /documents/{id}
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_any_permission, require_permission
from app.core.permissions import Perm
from app.database import get_db
from app.models.document import Document
from app.models.user import User
from app.schemas.rag import (
    IndexResponse,
    RagContextResponse,
    RagHit,
    RagSearchRequest,
    RagSearchResponse,
    RemoveResponse,
)
from app.services.rag_service import index_document, search
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/index-document",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
)
def rag_index_document(
    payload: dict = Body(..., examples=[{"document_id": "<uuid>"}]),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.RAG_INDEX)),
) -> IndexResponse:
    document_id = payload.get("document_id") if isinstance(payload, dict) else None
    if not isinstance(document_id, str) or not document_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="document_id (string) is required"
        )

    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        result = index_document(doc)
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Document file is missing on disk; re-upload required",
        )
    except Exception as e:
        # Surface a generic error to clients; full trace will be in server logs.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {type(e).__name__}",
        )

    doc.is_indexed = True
    doc.chunk_count = result.chunk_count
    db.commit()

    return IndexResponse(document_id=doc.id, chunk_count=result.chunk_count)


@router.delete("/remove-document/{document_id}", response_model=RemoveResponse)
def rag_remove_document(
    document_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.RAG_INDEX)),
) -> RemoveResponse:
    removed = get_vector_store().delete_document(document_id)

    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is not None:
        doc.is_indexed = False
        doc.chunk_count = 0
        db.commit()

    return RemoveResponse(document_id=document_id, removed_points=removed)


@router.post("/search", response_model=RagSearchResponse)
def rag_search(
    req: RagSearchRequest,
    current_user: User = Depends(require_permission(Perm.RAG_SEARCH)),
) -> RagSearchResponse:
    # Security: Clients without DOC_READ_ANY can only see hits from their own
    # company. We force the company_scope filter regardless of what the user
    # passed in the `company_name` field.
    company_scope: str | None = None
    if not current_user.has_permission(Perm.DOC_READ_ANY):
        if not current_user.company_name:
            # No company set => no visible documents.
            return RagSearchResponse(query=req.query, retrieved=0, reranked=0, results=[])
        company_scope = current_user.company_name

    try:
        hits, retrieved_count, reranked_count = search(
            query=req.query,
            top_k=req.top_k,
            company_name=req.company_name,
            document_type=req.document_type.value if req.document_type else None,
            document_id=req.document_id,
            company_scope=company_scope,
        )
    except Exception as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {type(e).__name__}",
        )

    return RagSearchResponse(
        query=req.query,
        retrieved=retrieved_count,
        reranked=reranked_count,
        results=[
            RagHit(
                document_id=h.document_id,
                chunk_index=h.chunk_index,
                text=h.text,
                score=h.score,
                title=h.title,
                company_name=h.company_name,
                document_type=h.document_type,  # type: ignore[arg-type]
            )
            for h in hits
        ],
    )


@router.get("/context/{document_id}", response_model=RagContextResponse)
def rag_context(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_any_permission(Perm.DOC_READ_ANY, Perm.DOC_READ_OWN_COMPANY)
    ),
) -> RagContextResponse:
    doc = db.query(Document).filter(Document.id == document_id).first()

    # Apply the same visibility rule as GET /documents/{id} — 404 on deny.
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    if not current_user.has_permission(Perm.DOC_READ_ANY):
        # Client-only path: must match their company.
        if (
            not current_user.company_name
            or current_user.company_name.strip().lower()
            != (doc.company_name or "").strip().lower()
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunks = get_vector_store().fetch_document_chunks(document_id, limit=200)
    return RagContextResponse(
        document_id=document_id,
        chunk_count=len(chunks),
        chunks=[
            RagHit(
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                score=0.0,
                title=c.payload.get("title", doc.title),
                company_name=c.payload.get("company_name", doc.company_name),
                document_type=c.payload.get("document_type", doc.document_type.value),  # type: ignore[arg-type]
            )
            for c in chunks
        ],
    )
