"""Idempotent bootstrap: seed permissions, default roles, and (optionally) the
first admin user.

This is invoked from `main.py` on startup. Running multiple times is safe — it
only inserts rows that don't already exist.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.permissions import ALL_PERMISSIONS, DEFAULT_ROLE_PERMISSIONS, Role as RoleConst
from app.core.security import hash_password
from app.models.role import Permission, Role
from app.models.user import User


def seed_defaults(db: Session) -> None:
    _seed_permissions(db)
    _seed_roles(db)
    _seed_admin_user(db)


def _seed_permissions(db: Session) -> None:
    existing_keys = {p.key for p in db.query(Permission).all()}
    new_rows = [
        Permission(id=str(uuid.uuid4()), key=k, description=k.replace(":", " ").title())
        for k in ALL_PERMISSIONS
        if k not in existing_keys
    ]
    if new_rows:
        db.add_all(new_rows)
        db.commit()


def _seed_roles(db: Session) -> None:
    perms_by_key = {p.key: p for p in db.query(Permission).all()}

    existing_roles = {r.name: r for r in db.query(Role).all()}
    for name, perm_keys in DEFAULT_ROLE_PERMISSIONS.items():
        wanted_perms = [perms_by_key[k] for k in perm_keys if k in perms_by_key]
        if name not in existing_roles:
            db.add(
                Role(
                    id=str(uuid.uuid4()),
                    name=name,
                    description=name.replace("_", " ").title(),
                    permissions=wanted_perms,
                )
            )
        else:
            # Ensure existing role has at least the default permissions
            role = existing_roles[name]
            current = {p.key for p in role.permissions}
            for k in perm_keys:
                if k not in current and k in perms_by_key:
                    role.permissions.append(perms_by_key[k])
    db.commit()


def _seed_admin_user(db: Session) -> None:
    settings = get_settings()
    # Only create the bootstrap admin if there are zero users in the system.
    if db.query(User).count() > 0:
        return

    admin_role = db.query(Role).filter(Role.name == RoleConst.ADMIN).first()
    if admin_role is None:
        return

    admin = User(
        id=str(uuid.uuid4()),
        email=settings.bootstrap_admin_email.lower(),
        username=settings.bootstrap_admin_username,
        hashed_password=hash_password(settings.bootstrap_admin_password),
        is_active=True,
    )
    admin.roles.append(admin_role)
    db.add(admin)
    db.commit()
