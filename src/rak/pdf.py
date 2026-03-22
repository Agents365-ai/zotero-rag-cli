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
