"""Embedding model wrapper with pluggable provider.

Two providers are supported, selected by the ``EMBEDDING_PROVIDER`` setting:

  * ``local_bge``     — sentence-transformers BGE bi-encoder, runs locally.
                       Requires huggingface.co reachable on first run to
                       download the model into ``~/.cache/huggingface``.
  * ``cisco_aiverse`` — calls Cisco's internal Aiverse gateway over OAuth2.

Both providers expose the same interface (``embed_passages`` /
``embed_query`` / ``dim``) so the rest of the RAG pipeline doesn't care which
one is active.

BGE convention: prepend a short instruction prefix to queries so the query and
passage representations live in the same semantic space. We apply the same
convention to the Cisco provider since most retrieval-tuned models (NV-EmbedQA,
BGE, E5) benefit from it.
"""

from __future__ import annotations

from threading import Lock
from typing import List, Protocol

from app.config import get_settings

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class _EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...
    def embed_passages(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, query: str) -> List[float]: ...


class _LocalBgeProvider:
    """sentence-transformers BGE bi-encoder, loaded lazily."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        s = get_settings()
        self._model = SentenceTransformer(s.embedding_model)
        self._dim = s.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        v = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return _to_list(v)

    def embed_query(self, query: str) -> List[float]:
        v = self._model.encode(
            [_QUERY_PREFIX + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _to_list(v)[0]


class _CiscoAiverseProvider:
    """Embeddings via Cisco Aiverse gateway. Network-only; no local model."""

    def __init__(self) -> None:
        from app.services.cisco_ai_client import get_client

        self._client = get_client()
        self._dim_declared = get_settings().embedding_dim
        self._dim_verified: int | None = None

    @property
    def dim(self) -> int:
        # We trust the configured value until the first embedding probe confirms
        # the real one. If they disagree at probe time we raise loudly so the
        # operator notices the misconfig before any vectors hit Qdrant.
        return self._dim_verified or self._dim_declared

    def _verify_dim(self, sample: List[float]) -> None:
        if self._dim_verified is not None:
            return
        actual = len(sample)
        if actual != self._dim_declared:
            raise RuntimeError(
                f"Cisco Aiverse embedding returned dim={actual} but "
                f"EMBEDDING_DIM={self._dim_declared}. Update EMBEDDING_DIM and "
                f"drop or rename the Qdrant collection so it gets recreated."
            )
        self._dim_verified = actual

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Batch to keep request payloads reasonable on the gateway.
        out: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self._client.embed(batch)
            if vecs:
                self._verify_dim(vecs[0])
            out.extend(_normalize_l2(v) for v in vecs)
        return out

    def embed_query(self, query: str) -> List[float]:
        v = self._client.embed_one(_QUERY_PREFIX + query)
        self._verify_dim(v)
        return _normalize_l2(v)


class EmbeddingModel:
    """Singleton facade with thread-safe lazy initialization."""

    _instance: "EmbeddingModel | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        provider = get_settings().embedding_provider
        if provider == "cisco_aiverse":
            self._impl: _EmbeddingProvider = _CiscoAiverseProvider()
        else:
            self._impl = _LocalBgeProvider()

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Forget the cached instance — used by tests / hot-reloads."""
        with cls._lock:
            cls._instance = None

    @property
    def dim(self) -> int:
        return self._impl.dim

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return self._impl.embed_passages(texts)

    def embed_query(self, query: str) -> List[float]:
        return self._impl.embed_query(query)


def _to_list(vectors) -> List[List[float]]:
    import numpy as np

    if isinstance(vectors, np.ndarray):
        return vectors.astype("float32").tolist()
    return [list(map(float, v)) for v in vectors]


def _normalize_l2(vec: List[float]) -> List[float]:
    """Project to the unit sphere so cosine == inner-product in Qdrant."""
    import math

    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return list(vec)
    inv = 1.0 / norm
    return [x * inv for x in vec]
