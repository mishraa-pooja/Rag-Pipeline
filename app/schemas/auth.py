"""Pydantic models for auth flows.

These schemas drive request validation. We use explicit allow-listed fields
(no `extra="allow"`, no automatic binding to ORM objects) — this prevents mass
assignment vulnerabilities where a user could send `is_active`, `roles`, etc.,
on registration and have them silently applied.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# Password constraints (OWASP-aligned): min 12 chars, allow Unicode, allow spaces,
# cap length at 128 to limit bcrypt's 72-byte truncation surprise and DoS.
_PWD_MIN = 12
_PWD_MAX = 128


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=_PWD_MIN, max_length=_PWD_MAX)
    company_name: str | None = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    """JSON login. Accepts either `email` or `username` in the `identifier` field."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=_PWD_MAX)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterResponse(BaseModel):
    id: str
    email: EmailStr
    username: str
    company_name: str | None = None
    roles: list[str]
