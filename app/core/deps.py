"""FastAPI dependencies: current user resolution and permission gates.

Authorization is enforced server-side on every protected endpoint via the
`require_permission(perm)` factory. We deny by default: any endpoint that does
not declare a permission dependency receives unauthenticated traffic only if
it intentionally omits `get_current_user`.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session, joinedload

from app.core.security import decode_access_token
from app.database import get_db
from app.models.user import User

# tokenUrl is informational for OpenAPI / Swagger UI. The actual login is at /auth/login.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


_CREDENTIALS_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the authenticated user from the bearer token. Raises 401 on failure."""
    if not token:
        raise _CREDENTIALS_EXC

    try:
        payload = decode_access_token(token)
    except JWTError:
        raise _CREDENTIALS_EXC

    user_id = payload.get("sub")
    if not user_id:
        raise _CREDENTIALS_EXC

    # Eager-load roles -> permissions so permission checks don't trigger N+1 queries
    # and so the user object can be safely accessed after the request scope closes.
    user = (
        db.query(User)
        .options(joinedload(User.roles))
        .filter(User.id == user_id)
        .first()
    )
    if user is None or not user.is_active:
        raise _CREDENTIALS_EXC
    return user


def require_permission(permission_key: str) -> Callable[[User], User]:
    """Return a dependency that ensures the current user has the given permission."""

    def _checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission(permission_key):
            # Generic 403 message; never reveal which permission is missing in a way
            # an unauthenticated user can probe.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return _checker


def require_any_permission(*permission_keys: str) -> Callable[[User], User]:
    """Return a dependency that requires at least one of the given permissions."""

    def _checker(current_user: User = Depends(get_current_user)) -> User:
        if not any(current_user.has_permission(p) for p in permission_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return _checker
