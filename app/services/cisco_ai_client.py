"""Client for Cisco's internal Aiverse AI gateway.

Authenticates with the IT-issued OAuth2 client-credentials flow (Duo SSO),
caches the bearer token until it nears expiry, and exposes two narrow methods
used by the RAG pipeline:

  * ``embed(texts)`` — OpenAI-compatible POST {embed_base}/embeddings
  * ``rerank(query, passages, top_k)`` — supports either Cohere-style
    ``/rerank`` or NVIDIA-NIM-style ``/ranking`` (auto-detected on first call).

Security:
- Credentials are read exclusively from environment variables; nothing is
  hardcoded. The validator in ``app.config`` refuses to start with placeholder
  values, and ``.env`` is gitignored.
- Token lifetime is honored from the ``expires_in`` claim with a 5-minute
  safety margin; we never reuse a token past its expiry.
- All HTTP failures surface as a single ``CiscoAIError`` so callers can return
  a generic 5xx without leaking gateway internals to API clients.
- The token and gateway responses are never written to the application log.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, List

import requests

from app.config import get_settings


class CiscoAIError(RuntimeError):
    """Wrapper for any failure talking to aiverse.cisco.com."""


@dataclass(frozen=True)
class _CachedToken:
    value: str
    expires_at: float  # epoch seconds


class CiscoAIClient:
    """Thread-safe singleton client for the Cisco Aiverse gateway."""

    _instance: "CiscoAIClient | None" = None
    _instance_lock = threading.Lock()

    # Pre-computed reranker-endpoint preference. We try the first one; on
    # 404/405/422 we fall through to the next and remember which one worked.
    _RERANK_PATHS: tuple[tuple[str, str], ...] = (
        ("cohere", "/rerank"),  # body: {query, documents, top_n}
        ("nim", "/ranking"),    # body: {query: {text}, passages: [{text}]}
    )

    def __init__(self) -> None:
        s = get_settings()
        self._client_id = s.cisco_client_id
        self._client_secret = s.cisco_client_secret
        self._token_url = s.cisco_token_url
        self._scope = s.cisco_token_scope or None

        self._embed_base = s.cisco_embedding_base_url.rstrip("/")
        self._embed_model = s.cisco_embedding_model
        self._rerank_base = s.cisco_rerank_base_url.rstrip("/")
        self._rerank_model = s.cisco_rerank_model

        self._timeout = s.cisco_request_timeout
        self._verify = True  # always validate TLS — let _ensure_ca_bundle pick the CA

        self._token_lock = threading.Lock()
        self._token: _CachedToken | None = None
        self._rerank_flavor: str | None = None  # learned on first successful call

    @classmethod
    def get(cls) -> "CiscoAIClient":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ auth

    def _bearer_token(self) -> str:
        """Return a valid OAuth2 bearer token, refreshing if near expiry."""
        now = time.monotonic()
        with self._token_lock:
            if self._token is not None and self._token.expires_at - 300 > now:
                return self._token.value

            payload: dict[str, str] = {"grant_type": "client_credentials"}
            if self._scope:
                payload["scope"] = self._scope

            try:
                resp = requests.post(
                    self._token_url,
                    data=payload,
                    auth=(self._client_id, self._client_secret),
                    timeout=self._timeout,
                    verify=self._verify,
                )
            except requests.RequestException as e:
                raise CiscoAIError(f"token endpoint unreachable: {type(e).__name__}") from e

            if resp.status_code != 200:
                # Don't echo the response body — it may contain internal details.
                raise CiscoAIError(
                    f"token endpoint returned HTTP {resp.status_code}"
                )

            try:
                data = resp.json()
            except ValueError as e:
                raise CiscoAIError("token endpoint returned non-JSON body") from e

            access = data.get("access_token")
            if not isinstance(access, str) or not access:
                raise CiscoAIError("token response missing access_token")

            expires_in = int(data.get("expires_in", 3600))
            self._token = _CachedToken(
                value=access,
                expires_at=now + max(expires_in, 60),
            )
            return access

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._bearer_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------ embeddings

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """OpenAI-compatible embeddings call. Returns one vector per input."""
        text_list = [t for t in texts if t]
        if not text_list:
            return []

        body: dict[str, Any] = {"input": text_list, "model": self._embed_model}
        url = f"{self._embed_base}/embeddings"

        try:
            resp = requests.post(
                url,
                json=body,
                headers=self._auth_headers(),
                timeout=self._timeout,
                verify=self._verify,
            )
        except requests.RequestException as e:
            raise CiscoAIError(f"embedding endpoint unreachable: {type(e).__name__}") from e

        if resp.status_code != 200:
            raise CiscoAIError(f"embedding endpoint returned HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as e:
            raise CiscoAIError("embedding endpoint returned non-JSON body") from e

        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise CiscoAIError("embedding response missing 'data' list")

        vectors: list[list[float]] = []
        for item in sorted(items, key=lambda x: int(x.get("index", 0))):
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise CiscoAIError("embedding response item missing 'embedding'")
            vectors.append([float(v) for v in vec])
        return vectors

    def embed_one(self, text: str) -> List[float]:
        out = self.embed([text])
        if not out:
            raise CiscoAIError("empty embedding result")
        return out[0]

    # ---------------------------------------------------------------- rerank

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int,
    ) -> List[tuple[int, float]]:
        """Return ``[(passage_index, score), ...]`` sorted by score desc.

        On first call the schema (Cohere vs NVIDIA NIM) is auto-detected and
        memoized. Indices are *into the passages list as provided*.
        """
        if not passages:
            return []

        # If we've already learned which path works, try it first; otherwise
        # iterate the preference order.
        order = list(self._RERANK_PATHS)
        if self._rerank_flavor:
            order.sort(key=lambda x: 0 if x[0] == self._rerank_flavor else 1)

        last_error: Exception | None = None
        for flavor, path in order:
            try:
                results = self._rerank_call(flavor, path, query, passages, top_k)
            except _RerankSchemaMismatch as e:
                last_error = e
                continue
            self._rerank_flavor = flavor
            return results

        raise CiscoAIError(
            f"reranker schema not understood (tried {[f for f, _ in self._RERANK_PATHS]})"
        ) from last_error

    def _rerank_call(
        self,
        flavor: str,
        path: str,
        query: str,
        passages: List[str],
        top_k: int,
    ) -> List[tuple[int, float]]:
        url = f"{self._rerank_base}{path}"
        if flavor == "cohere":
            body: dict[str, Any] = {
                "model": self._rerank_model,
                "query": query,
                "documents": passages,
                "top_n": top_k,
            }
        elif flavor == "nim":
            body = {
                "model": self._rerank_model,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
            }
        else:  # pragma: no cover — guarded by _RERANK_PATHS
            raise ValueError(f"unknown rerank flavor {flavor!r}")

        try:
            resp = requests.post(
                url,
                json=body,
                headers=self._auth_headers(),
                timeout=self._timeout,
                verify=self._verify,
            )
        except requests.RequestException as e:
            raise CiscoAIError(f"rerank endpoint unreachable: {type(e).__name__}") from e

        # Treat 404/405/422 as a schema mismatch — those are the standard
        # responses when a NIM endpoint receives a Cohere payload (or vice versa).
        if resp.status_code in (404, 405, 422):
            raise _RerankSchemaMismatch(f"flavor {flavor!r} got HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise CiscoAIError(f"rerank endpoint returned HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as e:
            raise CiscoAIError("rerank endpoint returned non-JSON body") from e

        if flavor == "cohere":
            items = data.get("results")
            if not isinstance(items, list):
                raise _RerankSchemaMismatch("cohere: no 'results' list")
            out = [
                (int(it["index"]), float(it["relevance_score"]))
                for it in items
                if "index" in it and "relevance_score" in it
            ]
        else:  # nim
            items = data.get("rankings")
            if not isinstance(items, list):
                raise _RerankSchemaMismatch("nim: no 'rankings' list")
            out = [
                (int(it["index"]), float(it.get("logit", it.get("score", 0.0))))
                for it in items
                if "index" in it
            ]

        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]


class _RerankSchemaMismatch(Exception):
    """Internal signal: try the next rerank flavor."""


def get_client() -> CiscoAIClient:
    return CiscoAIClient.get()
