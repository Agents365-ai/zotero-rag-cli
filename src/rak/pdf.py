from __future__ import annotations

from pathlib import Path


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def find_pdf(storage_dir: Path, item_key: str) -> Path | None:
    key_dir = storage_dir / item_key
    if not key_dir.is_dir():
        return None
    pdfs = list(key_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping word-level chunks.

    Returns a list of chunk strings. If the text fits in one chunk,
    returns a single-element list.
    """
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
