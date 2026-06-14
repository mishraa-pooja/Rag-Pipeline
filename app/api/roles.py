"""Role + user-role management endpoints.

All endpoints in this router are gated on the `role:manage` / `user:manage`
permissions, which only the Admin role holds by default.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.core.deps import require_permission
from app.core.permissions import ALL_PERMISSIONS, Perm
from app.database import get_db
from app.models.role import Permission, Role
from app.models.user import User
from app.schemas.role import AssignRoleRequest, RoleCreateRequest, RoleResponse
from app.schemas.user import (
    UserBrief,
    UserListResponse,
    UserPermissionsResponse,
    UserRolesResponse,
)

router = APIRouter(tags=["roles"])


@router.post(
    "/roles/create",
    response_model=RoleResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_role(
    req: RoleCreateRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.ROLE_MANAGE)),
) -> RoleResponse:
    if db.query(Role).filter(Role.name == req.name).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Role already exists")

    # Allow-list permissions: only keys that exist in the table can be attached.
    invalid = [p for p in req.permissions if p not in ALL_PERMISSIONS]
    if invalid:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown permission keys: {sorted(invalid)}",
        )

    perm_rows = []
    if req.permissions:
        perm_rows = db.query(Permission).filter(Permission.key.in_(req.permissions)).all()

    role = Role(
        id=str(uuid.uuid4()),
        name=req.name,
        description=req.description,
        permissions=perm_rows,
    )
    db.add(role)
    db.commit()
    db.refresh(role)

    return RoleResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        permissions=[p.key for p in role.permissions],
    )


@router.post(
    "/users/assign-role",
    response_model=UserRolesResponse,
)
def assign_role(
    req: AssignRoleRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.USER_MANAGE)),
) -> UserRolesResponse:
    user = db.query(User).options(joinedload(User.roles)).filter(User.id == req.user_id).first()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")

    role = db.query(Role).filter(Role.name == req.role_name).first()
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Role not found")

    if not user.has_role(role.name):
        user.roles.append(role)
        db.commit()
        db.refresh(user)

    return UserRolesResponse(user_id=user.id, roles=[r.name for r in user.roles])


@router.get("/users", response_model=UserListResponse)
def list_users(
    q: str | None = Query(
        default=None,
        max_length=255,
        description="Case-insensitive substring match against username OR email.",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10_000),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.USER_MANAGE)),
) -> UserListResponse:
    """List users (admin only). Useful for finding a user's id before calling
    /users/{user_id}/roles or /users/assign-role.
    """
    query = db.query(User).options(joinedload(User.roles))
    if q:
        # ilike works on SQLite (case-insensitive ASCII) and Postgres.
        # We pre-escape SQL LIKE wildcards in the user-supplied substring so a
        # search for "a%" doesn't unintentionally match everything.
        safe = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{safe}%"
        query = query.filter(
            or_(
                User.username.ilike(pattern, escape="\\"),
                User.email.ilike(pattern, escape="\\"),
            )
        )

    total = query.count()
    rows = (
        query.order_by(User.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = [
        UserBrief(
            id=u.id,
            email=u.email,
            username=u.username,
            company_name=u.company_name,
            is_active=u.is_active,
            created_at=u.created_at,
            roles=[r.name for r in u.roles],
        )
        for u in rows
    ]
    return UserListResponse(total=total, items=items)


@router.get("/users/{user_id}/roles", response_model=UserRolesResponse)
def get_user_roles(
    user_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.USER_MANAGE)),
) -> UserRolesResponse:
    user = db.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserRolesResponse(user_id=user.id, roles=[r.name for r in user.roles])


@router.get("/users/{user_id}/permissions", response_model=UserPermissionsResponse)
def get_user_permissions(
    user_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission(Perm.USER_MANAGE)),
) -> UserPermissionsResponse:
    user = db.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserPermissionsResponse(
        user_id=user.id,
        permissions=sorted(user.permission_keys()),
    )
