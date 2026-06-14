"""Qdrant vector store wrapper.

Design notes:
- One collection holds every chunk from every document; the `document_id` and
  `chunk_index` are stored on each point's payload so we can list/delete by
  document.
- We use **cosine** distance because our embeddings (BGE) are L2-normalized at
  encode time. With unit vectors cosine ≈ inner product.
- Each point uses a deterministic UUID5 derived from `document_id + chunk_index`,
  so re-indexing a document overwrites its previous points instead of duplicating.
- All Qdrant calls go through this single module so it is easy to swap to a
  different backend (FAISS / Chroma) later without touching the routers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings

# Deterministic namespace so the same (document_id, chunk_index) always maps to
# the same point id. The literal UUID below is just a randomly-chosen, fixed
# namespace and is *not* a secret.
_CHUNK_NAMESPACE = uuid.UUID("9b3c5a52-3a1f-4b88-bf36-1d4f8d6a9b21")


@dataclass
class ScoredChunk:
    document_id: str
    chunk_index: int
    text: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    _instance: "VectorStore | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        settings = get_settings()
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30.0,
            prefer_grpc=False,
        )
        self._collection = settings.qdrant_collection
        self._dim = settings.embedding_dim
        self._ensure_collection()

    @classmethod
    def get(cls) -> "VectorStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            # If the stored vector dim disagrees with our configured dim
            # (typically because the EMBEDDING_PROVIDER was switched), the
            # collection is unusable. We drop and recreate it — safe for dev,
            # but ALERT: this destroys the indexed corpus. For production,
            # name a new collection in QDRANT_COLLECTION instead.
            info = self._client.get_collection(self._collection)
            stored_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            if int(stored_dim) != int(self._dim):
                self._client.delete_collection(self._collection)
            else:
                return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
        )
        # Payload indexes speed up metadata filters.
        for field in ("document_id", "company_name", "document_type", "uploaded_by"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Index may already exist on a re-run.
                pass

    @staticmethod
    def _point_id(document_id: str, chunk_index: int) -> str:
        return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document_id}:{chunk_index}"))

    def upsert_chunks(
        self,
        document_id: str,
        chunks: List[tuple[int, str]],
        vectors: List[List[float]],
        base_payload: dict[str, Any],
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points: List[qm.PointStruct] = []
        for (idx, text), vec in zip(chunks, vectors):
            payload = {
                **base_payload,
                "document_id": document_id,
                "chunk_index": idx,
                "text": text,
            }
            points.append(
                qm.PointStruct(
                    id=self._point_id(document_id, idx),
                    vector=vec,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def delete_document(self, document_id: str) -> int:
        filt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="document_id",
                    match=qm.MatchValue(value=document_id),
                )
            ]
        )
        # count() so we can return a meaningful number
        try:
            before = self._client.count(
                collection_name=self._collection,
                count_filter=filt,
                exact=True,
            ).count
        except Exception:
            before = 0

        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=filt),
            wait=True,
        )
        return int(before)

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        company_name: str | None = None,
        document_type: str | None = None,
        document_id: str | None = None,
    ) -> List[ScoredChunk]:
        must: List[qm.FieldCondition] = []
        if company_name:
            must.append(
                qm.FieldCondition(
                    key="company_name", match=qm.MatchValue(value=company_name)
                )
            )
        if document_type:
            must.append(
                qm.FieldCondition(
                    key="document_type", match=qm.MatchValue(value=document_type)
                )
            )
        if document_id:
            must.append(
                qm.FieldCondition(
                    key="document_id", match=qm.MatchValue(value=document_id)
                )
            )

        filt = qm.Filter(must=must) if must else None

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filt,
            with_payload=True,
            with_vectors=False,
        )
        return [
            ScoredChunk(
                document_id=h.payload.get("document_id", ""),
                chunk_index=int(h.payload.get("chunk_index", 0)),
                text=h.payload.get("text", ""),
                score=float(h.score),
                payload=dict(h.payload),
            )
            for h in hits
            if h.payload is not None
        ]

    def fetch_document_chunks(self, document_id: str, limit: int = 200) -> List[ScoredChunk]:
        """Return all chunks for a document (no query vector, no score)."""
        filt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="document_id", match=qm.MatchValue(value=document_id)
                )
            ]
        )
        results: List[ScoredChunk] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=False,
                limit=min(100, limit - len(results)),
                offset=offset,
            )
            for p in points:
                if p.payload is None:
                    continue
                results.append(
                    ScoredChunk(
                        document_id=p.payload.get("document_id", ""),
                        chunk_index=int(p.payload.get("chunk_index", 0)),
                        text=p.payload.get("text", ""),
                        score=0.0,
                        payload=dict(p.payload),
                    )
                )
                if len(results) >= limit:
                    return sorted(results, key=lambda c: c.chunk_index)
            if offset is None or not points:
                break
        return sorted(results, key=lambda c: c.chunk_index)


def get_vector_store() -> VectorStore:
    return VectorStore.get()
