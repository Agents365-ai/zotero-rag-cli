from __future__ import annotations

import csv
from io import StringIO


def to_csv(results: list[dict]) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["key", "title", "score", "source"])
    for r in results:
        writer.writerow([r["key"], r["title"], round(r["score"], 4), r["source"]])
    return buf.getvalue()


ZOTERO_TO_BIBTEX = {
    "conferencePaper": "inproceedings",
    "bookSection": "incollection",
    "book": "book",
    "thesis": "phdthesis",
    "report": "techreport",
    "presentation": "misc",
    "manuscript": "unpublished",
}


def _bibtex_entry_type(item_type: str) -> str:
    return ZOTERO_TO_BIBTEX.get(item_type, "article")


def _escape_bibtex(value: str) -> str:
    """Escape special BibTeX characters in field values."""
    for char in ('\\', '{', '}', '&', '%', '#', '_', '~', '^'):
        value = value.replace(char, f'\\{char}')
    return value


def _extract_year(date: str) -> str:
    if not date:
        return ""
    return date[:4] if len(date) >= 4 else date


def to_bibtex(results: list[dict]) -> str:
    entries = []
    for r in results:
        entry_type = _bibtex_entry_type(r.get("item_type", ""))
        year = _extract_year(r.get("date", ""))
        title = _escape_bibtex(r['title'])
        authors = _escape_bibtex(r.get('authors', ''))
        entry = (
            f"@{entry_type}{{{r['key']},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year}}},\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)
