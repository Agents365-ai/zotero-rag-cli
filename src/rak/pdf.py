from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_via_pymupdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", pdf_path, exc)
        return ""


def _extract_via_mineru(pdf_path: Path) -> str | None:
    """Call MinerU CLI to parse a PDF and return the Markdown text.
    Returns None on any failure so the caller can fall back.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            result = subprocess.run(
                ["mineru", "-p", str(pdf_path), "-o", tmp_dir],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning("MinerU failed for %s: %s", pdf_path, result.stderr)
                return None
            stem = pdf_path.stem
            md_file = Path(tmp_dir) / stem / "auto" / f"{stem}.md"
            if md_file.exists():
                return md_file.read_text(encoding="utf-8").strip()
            md_files = list(Path(tmp_dir).rglob("*.md"))
            if md_files:
                return md_files[0].read_text(encoding="utf-8").strip()
            logger.warning("MinerU produced no markdown for %s", pdf_path)
            return None
        except Exception as exc:
            logger.warning("MinerU error for %s: %s", pdf_path, exc)
            return None


def _extract_via_docling(pdf_path: Path) -> str | None:
    """Call Docling CLI to parse a PDF and return the Markdown text.
    Returns None on any failure so the caller can fall back.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            result = subprocess.run(
                ["docling", str(pdf_path), "--output", tmp_dir, "--format", "md"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning("Docling failed for %s: %s", pdf_path, result.stderr)
                return None
            md_files = list(Path(tmp_dir).rglob("*.md"))
            if md_files:
                return md_files[0].read_text(encoding="utf-8").strip()
            logger.warning("Docling produced no markdown for %s", pdf_path)
            return None
        except Exception as exc:
            logger.warning("Docling error for %s: %s", pdf_path, exc)
            return None


def extract_pdf_text(pdf_path: Path, provider: str = "pymupdf") -> str:
    if not pdf_path.exists():
        return ""
    if provider in ("mineru", "docling"):
        extractor = _extract_via_mineru if provider == "mineru" else _extract_via_docling
        text = extractor(pdf_path)
        if text is not None:
            return text
        logger.warning("Falling back to PyMuPDF for %s", pdf_path)
    return _extract_via_pymupdf(pdf_path)


def extract_file_text(file_path: Path, provider: str = "pymupdf") -> str:
    """Extract text from a PDF or Markdown file."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(file_path, provider=provider)
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


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines and markdown headings."""
    import re
    # Split on double newlines or before markdown headings
    parts = re.split(r'\n\s*\n|(?=\n#{1,6}\s)', text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size word-level chunking with overlap (fallback for oversized paragraphs)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into chunks, preferring paragraph/section boundaries.

    Splits on double newlines and markdown headings first, then merges
    small consecutive paragraphs up to chunk_size words. Oversized
    paragraphs are split with word-level overlap. Returns a single-element
    list if the text fits in one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) <= 1:
        # No paragraph structure found, fall back to word-level chunking
        return _chunk_words(text, chunk_size, overlap)

    # Merge small paragraphs, split oversized ones
    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words > chunk_size:
            # Flush accumulated parts first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_words = 0
            # Split oversized paragraph with overlap
            chunks.extend(_chunk_words(para, chunk_size, overlap))
        elif current_words + para_words > chunk_size and current_parts:
            # Current buffer is full, flush it
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_words = para_words
        else:
            current_parts.append(para)
            current_words += para_words

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks
