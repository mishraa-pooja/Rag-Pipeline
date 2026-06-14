"""Document CRUD and metadata search.

Authorization rules:
- Upload     : `document:upload`
- List/Get/Search: `document:read_any` OR `document:read_own_company` (clients
  may only see documents whose `company_name` matches their `User.company_name`).
- Delete     : `document:delete` (admin) OR the original uploader for their own.

Search is metadata-only (SQL-side); semantic search lives at `/rag/search`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_any_permission, require_permission
from app.core.permissions import Perm
from app.database import get_db
from app.models.document import Document, DocumentType
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.file_storage import delete_stored_file, save_upload

router = APIRouter(prefix="/documents", tags=["documents"])


def _can_view_document(user: User, doc: Document) -> bool:
    if user.has_permission(Perm.DOC_READ_ANY):
        return True
    if user.has_permission(Perm.DOC_READ_OWN_COMPANY):
        # Compare case-insensitively; client must have a company_name set.
        return bool(
            user.company_name
            and user.company_name.strip().lower() == (doc.company_name or "").strip().lower()
        )
    return False


def _apply_visibility_filter(query, user: User):
    """Return a SQLAlchemy query scoped to documents the user may see."""
    if user.has_permission(Perm.DOC_READ_ANY):
        return query
    if user.has_permission(Perm.DOC_READ_OWN_COMPANY):
        if not user.company_name:
            # Client without a company can't see any documents.
            return query.filter(Document.id.is_(None))  # always-false
        return query.filter(Document.company_name == user.company_name)
    # No read permission at all — return an always-false query.
    return query.filter(Document.id.is_(None))


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    title: str = Form(..., min_length=1, max_length=255),
    company_name: str = Form(..., min_length=1, max_length=255),
    document_type: DocumentType = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Perm.DOC_UPLOAD)),
) -> DocumentResponse:
    stored = save_upload(file)

    doc = Document(
        id=str(uuid.uuid4()),
        title=title.strip(),
        company_name=company_name.strip(),
        document_type=document_type,
        uploaded_by=current_user.id,
        storage_filename=stored.storage_filename,
        original_filename=stored.original_filename,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        sha256_hex=stored.sha256_hex,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return DocumentResponse.model_validate(doc)


# IMPORTANT: /documents/search must be declared BEFORE /documents/{document_id}
# so FastAPI doesn't try to match "search" as a document_id path parameter.
@router.get("/search", response_model=DocumentListResponse)
def search_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_any_permission(Perm.DOC_READ_ANY, Perm.DOC_READ_OWN_COMPANY)
    ),
    title: str | None = Query(default=None, max_length=255),
    company_name: str | None = Query(default=None, max_length=255),
    document_type: DocumentType | None = Query(default=None),
    uploaded_by: str | None = Query(default=None, max_length=36),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10_000),
) -> DocumentListResponse:
    q = db.query(Document)
    q = _apply_visibility_filter(q, current_user)

    if title:
        # LIKE patterns are safe via SQLAlchemy parameterization.
        q = q.filter(Document.title.ilike(f"%{title}%"))
    if company_name:
        q = q.filter(Document.company_name.ilike(f"%{company_name}%"))
    if document_type is not None:
        q = q.filter(Document.document_type == document_type)
    if uploaded_by:
        q = q.filter(Document.uploaded_by == uploaded_by)

    total = q.count()
    items = q.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    return DocumentListResponse(
        total=total,
        items=[DocumentResponse.model_validate(d) for d in items],
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_any_permission(Perm.DOC_READ_ANY, Perm.DOC_READ_OWN_COMPANY)
    ),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10_000),
) -> DocumentListResponse:
    q = _apply_visibility_filter(db.query(Document), current_user)
    total = q.count()
    items = q.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    return DocumentListResponse(
        total=total,
        items=[DocumentResponse.model_validate(d) for d in items],
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_any_permission(Perm.DOC_READ_ANY, Perm.DOC_READ_OWN_COMPANY)
    ),
) -> DocumentResponse:
    doc = db.query(Document).filter(Document.id == document_id).first()
    # Return 404 for both "not found" and "no access" to avoid revealing existence (IDOR).
    if doc is None or not _can_view_document(current_user, doc):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    doc = db.query(Document).filter(Document.id == document_id).first()
    # Same 404-on-deny pattern.
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Allowed if: explicit DOC_DELETE permission, OR uploader of this doc.
    if not (
        current_user.has_permission(Perm.DOC_DELETE)
        or doc.uploaded_by == current_user.id
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    storage_name = doc.storage_filename
    doc_id = doc.id

    # Best-effort vector cleanup. Imported lazily so the router stays importable
    # even when Qdrant isn't reachable.
    try:
        from app.services.vector_store import get_vector_store

        get_vector_store().delete_document(doc_id)
    except Exception:
        # Don't block file/DB deletion on vector-store failures; log in real systems.
        pass

    db.delete(doc)
    db.commit()

    # Delete the on-disk file last so a DB rollback doesn't orphan a deleted file.
    delete_stored_file(storage_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
