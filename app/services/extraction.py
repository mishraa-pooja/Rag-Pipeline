"""Text extraction from supported document types.

Supported MIME types:
- application/pdf  -> pypdf
- text/plain       -> direct decode
- DOCX             -> python-docx

Defensive controls:
- We *never* construct file paths from user input here; we receive an absolute
  path that has already been path-containment-checked at upload time.
- pypdf is used in pure read mode; we do not evaluate scripts/forms.
- We cap the extracted text length to bound memory and embedding cost.
"""

from __future__ import annotations

from pathlib import Path

# Cap extracted text to keep memory + indexing time bounded. Documents larger
# than this are still readable but only the first MAX_CHARS are indexed.
MAX_CHARS = 1_500_000


def extract_text(path: str, content_type: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Document file not found on disk: {path}")

    ct = (content_type or "").lower()
    if ct == "application/pdf":
        text = _extract_pdf(p)
    elif ct == "text/plain":
        text = _extract_text(p)
    elif ct == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        text = _extract_docx(p)
    else:
        raise ValueError(f"Unsupported content type for extraction: {content_type}")

    text = _normalize(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


def _extract_pdf(p: Path) -> str:
    from pypdf import PdfReader

    parts: list[str] = []
    reader = PdfReader(str(p))
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # A malformed page should not abort the whole document.
            parts.append("")
    return "\n".join(parts)


def _extract_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _extract_docx(p: Path) -> str:
    import docx  # python-docx

    doc = docx.Document(str(p))
    return "\n".join(par.text for par in doc.paragraphs)


def _normalize(text: str) -> str:
    # Collapse pathological whitespace runs but keep paragraph structure.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
