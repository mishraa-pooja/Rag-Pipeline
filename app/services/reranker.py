"""Pluggable cross-encoder reranker.

Two providers, selected by ``EMBEDDING_PROVIDER`` for symmetry with embeddings:

  * ``local_bge``     — sentence-transformers CrossEncoder, runs locally.
  * ``cisco_aiverse`` — calls Cisco's Aiverse reranking-service.

If ``CISCO_RERANK_ENABLED=false`` we skip the reranker entirely and the
pipeline returns the top-K vector-search results as-is, preserving the
embedding's initial score. Useful when only the embedding service is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import List

from app.config import get_settings


@dataclass
class Candidate:
    text: str
    payload: dict
    initial_score: float


class _LocalBgeReranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(get_settings().reranker_model)

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: int,
    ) -> List[tuple[Candidate, float]]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class _CiscoAiverseReranker:
    def __init__(self) -> None:
        from app.services.cisco_ai_client import get_client

        self._client = get_client()

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: int,
    ) -> List[tuple[Candidate, float]]:
        if not candidates:
            return []
        passages = [c.text for c in candidates]
        ranked = self._client.rerank(query=query, passages=passages, top_k=top_k)
        return [(candidates[idx], score) for idx, score in ranked]


class _NoopReranker:
    """Identity reranker: keep the original retrieval order and scores."""

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: int,
    ) -> List[tuple[Candidate, float]]:
        return [(c, c.initial_score) for c in candidates[:top_k]]


class Reranker:
    """Singleton facade with thread-safe lazy initialization."""

    _instance: "Reranker | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        s = get_settings()
        if s.embedding_provider == "cisco_aiverse":
            self._impl = _NoopReranker() if not s.cisco_rerank_enabled else _CiscoAiverseReranker()
        else:
            self._impl = _LocalBgeReranker()

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: int,
    ) -> List[tuple[Candidate, float]]:
        return self._impl.rerank(query, candidates, top_k)
