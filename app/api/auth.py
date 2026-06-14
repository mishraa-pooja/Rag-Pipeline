"""Auth router: registration + login.

Security choices:
- Registration auto-assigns the `client` role only. Privilege escalation
  requires an admin to call /users/assign-role — preventing self-elevation.
- Login returns a **generic** error for both unknown user and wrong password,
  with consistent timing (we always run `verify_password` against either the
  real hash or a fixed dummy hash) to mitigate username enumeration.
- Passwords are bcrypt-hashed via the security module.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.permissions import Role as RoleConst
from app.core.security import create_access_token, hash_password, verify_password
from app.database import get_db
from app.models.role import Role
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# A bcrypt hash used purely for timing-equalization on the "unknown user"
# branch of login. Verifying against this hash takes the same wall-clock time
# as verifying against a real hash, so attackers cannot enumerate accounts via
# response timing. We compute it lazily on first use so the import is cheap
# and the value is guaranteed-valid for passlib.
_DUMMY_HASH: str | None = None


def _get_dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("__timing_equalization_only__")
    return _DUMMY_HASH


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    # Case-insensitive uniqueness check to avoid `Alice` / `alice` collisions.
    existing = (
        db.query(User)
        .filter(
            or_(
                func.lower(User.email) == req.email.lower(),
                func.lower(User.username) == req.username.lower(),
            )
        )
        .first()
    )
    if existing:
        # Generic message — do not reveal which field collided.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account with these credentials already exists",
        )

    client_role = db.query(Role).filter(Role.name == RoleConst.CLIENT).first()
    if client_role is None:
        # Should never happen after seed_defaults() runs.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: default client role missing",
        )

    user = User(
        id=str(uuid.uuid4()),
        email=req.email.lower(),
        username=req.username,
        hashed_password=hash_password(req.password),
        company_name=req.company_name,
        is_active=True,
    )
    user.roles.append(client_role)
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(
        id=user.id,
        email=user.email,  # type: ignore[arg-type]
        username=user.username,
        company_name=user.company_name,
        roles=[r.name for r in user.roles],
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    settings = get_settings()

    ident = req.identifier.strip().lower()
    user = (
        db.query(User)
        .filter(or_(func.lower(User.email) == ident, func.lower(User.username) == ident))
        .first()
    )

    # Always run a bcrypt verify even when the user does not exist, against a
    # constant dummy hash, so the unknown-user path takes the same time as the
    # known-user path. We then branch on the boolean result.
    if user is None:
        verify_password(req.password, _get_dummy_hash())
        valid = False
    else:
        valid = verify_password(req.password, user.hashed_password) and user.is_active

    if not valid or user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.id, extra_claims={"username": user.username})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )
