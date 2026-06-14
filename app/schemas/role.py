"""Role / permission management schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RoleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str | None = Field(default=None, max_length=255)
    permissions: list[str] = Field(default_factory=list)


class RoleResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    permissions: list[str]


class AssignRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=36)
    role_name: str = Field(min_length=2, max_length=80)
