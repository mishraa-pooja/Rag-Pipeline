"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import auth as auth_router
from app.api import documents as documents_router
from app.api import rag as rag_router
from app.api import roles as roles_router
from app.config import get_settings
from app.database import Base, SessionLocal, engine
from app.seed import seed_defaults

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables (sqlite-friendly bootstrap; switch to Alembic for prod).
    Base.metadata.create_all(bind=engine)

    # Seed default permissions, roles, and the bootstrap admin if needed.
    with SessionLocal() as db:
        try:
            seed_defaults(db)
        except Exception as e:
            logger.exception("Seed failed: %s", e)
            raise

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "FastAPI service for storing, managing and semantically searching "
            "financial documents using a RAG pipeline (BGE embeddings + Qdrant + "
            "BGE reranker)."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS: keep restrictive by default. For browser SPAs, configure ALLOWED_ORIGINS
    # via env in a follow-up. We use a closed list to avoid wildcard origins, which
    # would weaken CSRF defenses and cookie isolation if cookies were ever used.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/health", tags=["meta"])
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    app.include_router(auth_router.router)
    app.include_router(roles_router.router)
    app.include_router(documents_router.router)
    app.include_router(rag_router.router)

    return app


app = create_app()
