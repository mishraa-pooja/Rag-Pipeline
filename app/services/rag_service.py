"""High-level RAG orchestration: index a document and run retrieval + reranking."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.models.document import Document
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingModel
from app.services.extraction import extract_text
from app.services.reranker import Candidate, Reranker
from app.services.vector_store import ScoredChunk, get_vector_store


@dataclass
class IndexResult:
    document_id: str
    chunk_count: int


def index_document(document: Document) -> IndexResult:
    """Extract → chunk → embed → upsert into Qdrant. Returns the chunk count."""
    settings = get_settings()
    upload_dir = settings.upload_dir_path
    absolute_path = str((upload_dir / document.storage_filename).resolve())

    text = extract_text(absolute_path, document.content_type)
    chunks = chunk_text(text)
    if not chunks:
        return IndexResult(document_id=document.id, chunk_count=0)

    embed_model = EmbeddingModel.get()
    vectors = embed_model.embed_passages([c.text for c in chunks])

    store = get_vector_store()
    # Replace any previous embeddings for this document by deleting first.
    store.delete_document(document.id)
    base_payload = {
        "title": document.title,
        "company_name": document.company_name,
        "document_type": document.document_type.value,
        "uploaded_by": document.uploaded_by,
        "created_at": document.created_at.isoformat() if document.created_at else None,
    }
    written = store.upsert_chunks(
        document_id=document.id,
        chunks=[(c.index, c.text) for c in chunks],
        vectors=vectors,
        base_payload=base_payload,
    )
    return IndexResult(document_id=document.id, chunk_count=written)


@dataclass
class RetrievedChunk:
    document_id: str
    chunk_index: int
    text: str
    score: float
    title: str
    company_name: str
    document_type: str


def search(
    query: str,
    top_k: int,
    company_name: str | None = None,
    document_type: str | None = None,
    document_id: str | None = None,
    company_scope: str | None = None,
) -> tuple[list[RetrievedChunk], int, int]:
    """Run the two-stage retrieve→rerank pipeline.

    Returns: (final_results, retrieved_count, reranked_count).

    `company_scope` is a *security* filter applied for Client-role users so they
    only ever see hits from their own company, regardless of what they passed in
    `company_name`.
    """
    settings = get_settings()
    embed_model = EmbeddingModel.get()
    qvec = embed_model.embed_query(query)

    effective_company = company_scope if company_scope is not None else company_name

    raw_hits: list[ScoredChunk] = get_vector_store().search(
        query_vector=qvec,
        top_k=settings.retrieve_top_k,
        company_name=effective_company,
        document_type=document_type,
        document_id=document_id,
    )

    candidates = [
        Candidate(text=h.text, payload=h.payload, initial_score=h.score) for h in raw_hits
    ]
    reranked = Reranker.get().rerank(query, candidates, top_k=min(top_k, settings.rerank_top_k))

    results: list[RetrievedChunk] = []
    for cand, score in reranked:
        p = cand.payload
        results.append(
            RetrievedChunk(
                document_id=p.get("document_id", ""),
                chunk_index=int(p.get("chunk_index", 0)),
                text=p.get("text", ""),
                score=float(score),
                title=p.get("title", ""),
                company_name=p.get("company_name", ""),
                document_type=p.get("document_type", "other"),
            )
        )
    return results, len(raw_hits), len(reranked)
