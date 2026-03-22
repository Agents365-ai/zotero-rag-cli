from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", pdf_path, exc)
        return ""


def extract_file_text(file_path: Path) -> str:
    """Extract text from a PDF or Markdown file."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(file_path)
    if suffix == ".md":
        try:
            return file_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return ""
    return ""


def find_attachments(storage_dir: Path, item_key: str) -> list[Path]:
    """Find all PDF and Markdown files in a Zotero item's storage directory."""
    key_dir = storage_dir / item_key
    if not key_dir.is_dir():
        return []
    files = list(key_dir.glob("*.pdf")) + list(key_dir.glob("*.md"))
    return sorted(files)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping word-level chunks.

    Returns a list of chunk strings. If the text fits in one chunk,
    returns a single-element list.
    """
    if overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks
