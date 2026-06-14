"""User model with the user↔role association table."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.role import Role


user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", String(36), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    roles: Mapped[List["Role"]] = relationship(
        "Role",
        secondary=user_roles,
        back_populates="users",
        lazy="selectin",
    )

    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="uploader",
        cascade="all, delete-orphan",
    )

    def permission_keys(self) -> set[str]:
        """Flattened set of permission keys granted via the user's roles."""
        result: set[str] = set()
        for role in self.roles:
            for perm in role.permissions:
                result.add(perm.key)
        return result

    def has_permission(self, key: str) -> bool:
        return key in self.permission_keys()

    def has_role(self, role_name: str) -> bool:
        return any(r.name == role_name for r in self.roles)
