"""SQLAlchemy engine, session factory and declarative base.

We use SQLAlchemy ORM with parameterized queries everywhere — no raw SQL string
concatenation with user input — which is the primary defense against SQL
injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _build_engine():
    settings = get_settings()
    url = settings.database_url

    if url.startswith("sqlite"):
        # Ensure parent directory exists for file-backed sqlite URLs.
        # URL form: sqlite:///./data/app.db
        path_part = url.split("sqlite:///", 1)[-1]
        Path(path_part).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is required for FastAPI's threaded request handling
        # with a single sqlite file. We rely on per-request Session scope to keep
        # concurrent access safe.
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            future=True,
        )

    return create_engine(url, future=True, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
