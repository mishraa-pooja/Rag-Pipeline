"""User-related response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class UserBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    username: str
    company_name: str | None = None
    is_active: bool
    created_at: datetime
    roles: list[str] = []


class UserListResponse(BaseModel):
    total: int
    items: list[UserBrief]


class UserRolesResponse(BaseModel):
    user_id: str
    roles: list[str]


class UserPermissionsResponse(BaseModel):
    user_id: str
    permissions: list[str]
