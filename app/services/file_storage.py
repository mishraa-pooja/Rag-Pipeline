"""Secure file upload storage.

Defense-in-depth controls implemented here:

1. *Server-generated names*: the on-disk filename is a UUID; the user-supplied
   filename is stored only as metadata for display.
2. *Path containment*: the resolved path is required to be inside
   `settings.upload_dir`, blocking traversal via `..` or absolute names.
3. *MIME allow-list*: only `application/pdf`, `text/plain`, and the DOCX MIME
   are accepted (configurable via env).
4. *Magic-byte sniffing*: declared content type must agree with the file's
   actual signature for PDF and DOCX (DOCX is a ZIP); plain text is checked
   for absence of NUL bytes.
5. *Size cap*: enforced both via streaming read with a hard byte counter and
   via the configured `MAX_UPLOAD_BYTES`.
6. *Integrity hash*: SHA-256 of the stored bytes is recorded so tampering with
   the on-disk file can be detected.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

_MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

_PDF_MAGIC = b"%PDF-"
# DOCX is a ZIP container; first 4 bytes are "PK\x03\x04".
_ZIP_MAGIC = b"PK\x03\x04"


@dataclass
class StoredFile:
    storage_filename: str  # the basename on disk (UUID + ext)
    absolute_path: str
    size_bytes: int
    sha256_hex: str
    content_type: str
    original_filename: str


def _safe_storage_name(content_type: str) -> str:
    ext = _MIME_TO_EXT.get(content_type, "")
    # 32 hex chars = 128 bits of entropy. Independent of any user input.
    return f"{secrets.token_hex(16)}{ext}"


def _safe_original_filename(name: str | None) -> str:
    """Sanitize the *display* filename. The actual stored filename is server-generated."""
    if not name:
        return "upload"
    # Reject NULs and control characters; strip path separators.
    cleaned = "".join(c for c in name if c.isprintable() and c not in ("/", "\\"))
    cleaned = cleaned.strip()
    # Disallow leading dot/hyphen/space which can confuse shell tooling later.
    cleaned = cleaned.lstrip(". -")
    if not cleaned:
        cleaned = "upload"
    # Cap length; this is only metadata.
    return cleaned[:255]


def _validate_magic_bytes(content_type: str, head: bytes) -> None:
    if content_type == "application/pdf":
        if not head.startswith(_PDF_MAGIC):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="File content does not match PDF signature",
            )
    elif content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        if not head.startswith(_ZIP_MAGIC):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="File content does not match DOCX signature",
            )
    elif content_type == "text/plain":
        if b"\x00" in head:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="Text file must not contain NUL bytes",
            )


def save_upload(upload: UploadFile) -> StoredFile:
    """Validate and persist an UploadFile. Raises HTTPException on any check failure."""
    settings = get_settings()

    ctype = (upload.content_type or "").lower().strip()
    if ctype not in settings.allowed_upload_mimes:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {ctype or '<missing>'}",
        )

    target_dir = settings.upload_dir_path
    storage_name = _safe_storage_name(ctype)
    target_path = (target_dir / storage_name).resolve()

    # Path-containment check (defense in depth — storage_name is already safe).
    try:
        target_path.relative_to(target_dir)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid storage path")

    max_bytes = settings.max_upload_bytes
    digest = hashlib.sha256()
    bytes_written = 0
    first_chunk: bytes = b""

    # Stream to disk in 64 KiB chunks; abort if we exceed the size cap.
    fh: BinaryIO
    with open(target_path, "wb") as fh:
        try:
            while True:
                chunk = upload.file.read(64 * 1024)
                if not chunk:
                    break
                if bytes_written + len(chunk) > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {max_bytes} bytes",
                    )
                if not first_chunk:
                    first_chunk = chunk[:16]
                    _validate_magic_bytes(ctype, first_chunk)
                fh.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
        except HTTPException:
            # On any failure, remove the partial file.
            fh.close()
            try:
                os.remove(target_path)
            except OSError:
                pass
            raise

    if bytes_written == 0:
        try:
            os.remove(target_path)
        except OSError:
            pass
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Empty file")

    return StoredFile(
        storage_filename=storage_name,
        absolute_path=str(target_path),
        size_bytes=bytes_written,
        sha256_hex=digest.hexdigest(),
        content_type=ctype,
        original_filename=_safe_original_filename(upload.filename),
    )


def delete_stored_file(storage_filename: str) -> bool:
    """Delete a previously stored file. Returns True iff a file was removed.

    `storage_filename` is the basename produced by `_safe_storage_name`. We
    re-validate path containment before unlinking.
    """
    settings = get_settings()
    target_dir = settings.upload_dir_path
    candidate = (target_dir / storage_filename).resolve()
    try:
        candidate.relative_to(target_dir)
    except ValueError:
        # Refuse to follow paths outside the upload dir.
        return False
    if not candidate.exists() or not candidate.is_file():
        return False
    candidate.unlink()
    return True


def read_stored_bytes(storage_filename: str) -> bytes:
    """Read a stored file's bytes, with path-containment validation."""
    settings = get_settings()
    target_dir = settings.upload_dir_path
    candidate = (target_dir / storage_filename).resolve()
    candidate.relative_to(target_dir)  # raises ValueError on traversal
    return Path(candidate).read_bytes()
