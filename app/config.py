"""Application configuration loaded from environment variables / .env file.

All secrets (JWT keys, DB passwords, Qdrant API keys, etc.) come from the
environment. Nothing sensitive is hardcoded — see `.env.example` for shape.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Financial Document Management"
    app_env: str = Field(default="development")
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    jwt_secret_key: str = Field(
        ...,
        description="Long random string. Generate with `python -c 'import secrets; print(secrets.token_urlsafe(64))'`.",
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_issuer: str = "fin-doc-mgmt"
    jwt_audience: str = "fin-doc-mgmt-clients"

    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = Field(
        ...,
        description="Initial admin password; must be supplied via env. Used only when no users exist yet.",
    )

    database_url: str = "sqlite:///./data/app.db"

    upload_dir: str = "./data/uploads"
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MiB
    allowed_upload_mimes_raw: str = Field(
        default=(
            "application/pdf,text/plain,"
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        alias="ALLOWED_UPLOAD_MIMES",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "financial_documents"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_dim: int = 384

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieve_top_k: int = 20
    rerank_top_k: int = 5

    # --- Provider switch ---------------------------------------------------
    # "local_bge"    — use local sentence-transformers BGE (default; needs
    #                  HuggingFace reachable on first run to download models).
    # "cisco_aiverse" — call Cisco's internal Aiverse gateway over OAuth2.
    embedding_provider: str = "local_bge"

    # --- Cisco Aiverse settings -------------------------------------------
    # All read from env; never hardcoded. Required only when
    # EMBEDDING_PROVIDER=cisco_aiverse.
    #
    # Defaults below mirror the Cisco AIverse wiki for embeddinggemma-300m
    # and granite-reranker. The OAuth scope MUST be "read write" — using
    # just "write" returns 401 "Invalid user key in JWT token" even with a
    # valid token. Both models require FMSConsumer access provisioned via
    # the CPP Console; without it the same 401 message is returned (instead
    # of the documented 403), which can be misleading when debugging.
    cisco_client_id: str = ""
    cisco_client_secret: str = ""
    cisco_token_url: str = ""
    cisco_token_scope: str = "read write"
    cisco_embedding_base_url: str = "https://aiverse.cisco.com/embeddinggemma-300m/v1"
    cisco_embedding_model: str = "google/embeddinggemma-300m"
    cisco_rerank_base_url: str = "https://aiverse.cisco.com/granite-reranker/v1"
    cisco_rerank_model: str = "ibm-granite/granite-embedding-reranker-english-r2"
    cisco_request_timeout: int = 60
    # Set to false only if your reranker isn't deployed; retrieval-only.
    cisco_rerank_enabled: bool = True

    @field_validator("jwt_secret_key")
    @classmethod
    def _validate_jwt_secret(cls, v: str) -> str:
        if not v or len(v) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        if v == "replace-me-with-a-long-random-string":
            raise ValueError("JWT_SECRET_KEY is still the example value — set a real secret.")
        return v

    @field_validator("bootstrap_admin_password")
    @classmethod
    def _validate_admin_password(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError("BOOTSTRAP_ADMIN_PASSWORD must be at least 12 characters.")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"local_bge", "cisco_aiverse"}:
            raise ValueError(
                "EMBEDDING_PROVIDER must be one of: local_bge, cisco_aiverse"
            )
        return v

    @model_validator(mode="after")
    def _validate_cisco_provider(self) -> "Settings":
        if self.embedding_provider == "cisco_aiverse":
            missing = [
                name
                for name, value in {
                    "CISCO_CLIENT_ID": self.cisco_client_id,
                    "CISCO_CLIENT_SECRET": self.cisco_client_secret,
                    "CISCO_TOKEN_URL": self.cisco_token_url,
                    "CISCO_EMBEDDING_BASE_URL": self.cisco_embedding_base_url,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(
                    "EMBEDDING_PROVIDER=cisco_aiverse but these env vars are unset: "
                    + ", ".join(missing)
                )
        return self

    @property
    def allowed_upload_mimes(self) -> List[str]:
        return [m.strip().lower() for m in self.allowed_upload_mimes_raw.split(",") if m.strip()]

    @property
    def upload_dir_path(self) -> Path:
        p = Path(self.upload_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — single instance per process."""
    return Settings()  # type: ignore[call-arg]
