"""Authoritative list of permission keys and the default role -> permissions map.

The system implements an RBAC model:
    User -*- Role -*- Permission

These constants are seeded at startup so the database is the source of truth at
runtime, but the codebase always uses these string constants to *check* for a
permission. That way a misspelling is a static error rather than a silent
authorization bypass.
"""

from __future__ import annotations

from typing import Final


class Perm:
    """Permission keys used across the API. Keep in sync with the seed loader."""

    DOC_UPLOAD: Final[str] = "document:upload"
    DOC_READ_ANY: Final[str] = "document:read_any"
    DOC_READ_OWN_COMPANY: Final[str] = "document:read_own_company"
    DOC_EDIT: Final[str] = "document:edit"
    DOC_DELETE: Final[str] = "document:delete"

    ROLE_MANAGE: Final[str] = "role:manage"
    USER_MANAGE: Final[str] = "user:manage"

    RAG_INDEX: Final[str] = "rag:index"
    RAG_SEARCH: Final[str] = "rag:search"


ALL_PERMISSIONS: Final[tuple[str, ...]] = (
    Perm.DOC_UPLOAD,
    Perm.DOC_READ_ANY,
    Perm.DOC_READ_OWN_COMPANY,
    Perm.DOC_EDIT,
    Perm.DOC_DELETE,
    Perm.ROLE_MANAGE,
    Perm.USER_MANAGE,
    Perm.RAG_INDEX,
    Perm.RAG_SEARCH,
)


# Default roles defined in the assignment.
class Role:
    ADMIN: Final[str] = "admin"
    ANALYST: Final[str] = "financial_analyst"
    AUDITOR: Final[str] = "auditor"
    CLIENT: Final[str] = "client"


DEFAULT_ROLE_PERMISSIONS: Final[dict[str, tuple[str, ...]]] = {
    Role.ADMIN: ALL_PERMISSIONS,
    Role.ANALYST: (
        Perm.DOC_UPLOAD,
        Perm.DOC_READ_ANY,
        Perm.DOC_EDIT,
        Perm.RAG_INDEX,
        Perm.RAG_SEARCH,
    ),
    Role.AUDITOR: (
        Perm.DOC_READ_ANY,
        Perm.RAG_SEARCH,
    ),
    Role.CLIENT: (
        Perm.DOC_READ_OWN_COMPANY,
        Perm.RAG_SEARCH,
    ),
}
