"""Authentication primitives: password hashing and JWT issuance / verification.

Design notes (security-focused):
- Passwords are hashed with bcrypt via passlib. Bcrypt is on the approved list
  for password storage and includes a per-user salt.
- JWT algorithm is **pinned** (HS256 by default) and verification refuses to
  accept any other algorithm. We also validate `iss` and `aud` claims.
- We never log or surface the raw token contents, hashed password, or secret.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

# bcrypt is OWASP-acceptable for password storage. We pin to a single scheme so
# that downgrade attacks are not possible at verify time.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Return a bcrypt hash for the given plaintext password."""
    if not isinstance(plain_password, str) or not plain_password:
        raise ValueError("password must be a non-empty string")
    # bcrypt has a 72-byte input limit. Truncate defensively after warning the
    # caller in the auth router that long passwords are accepted but only the
    # first 72 bytes are significant. We do NOT pre-hash with SHA before bcrypt
    # because that subtly changes the security analysis; instead we cap input
    # length at the API boundary (see schemas).
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time-ish verification of a plaintext against a bcrypt hash."""
    try:
        return _pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # Any exception (e.g., malformed hash) is treated as a failed verify
        # without leaking which case occurred.
        return False


def create_access_token(
    subject: str,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Issue a signed JWT for the given subject (user id or username).

    Includes `iss`, `aud`, `iat`, `nbf`, `exp`. Algorithm is pinned via settings.
    """
    settings = get_settings()
    now = datetime.now(tz=timezone.utc)
    expire = now + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode: dict[str, Any] = {
        "sub": str(subject),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        # Never allow extra claims to overwrite the protected ones above.
        for k, v in extra_claims.items():
            if k in to_encode:
                continue
            to_encode[k] = v

    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT. Raises `JWTError` on any failure."""
    settings = get_settings()
    # Explicitly pin algorithms — refuse alg=none and any other algorithm.
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
        options={"require": ["exp", "iat", "sub", "iss", "aud"]},
    )


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "JWTError",
]
