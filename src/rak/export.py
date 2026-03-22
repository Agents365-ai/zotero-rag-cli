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


def to_bibtex(results: list[dict]) -> str:
    entries = []
    for r in results:
        entry = (
            f"@article{{{r['key']},\n"
            f"  title = {{{r['title']}}},\n"
            f"  author = {{{r.get('authors', '')}}},\n"
            f"  year = {{{r.get('date', '')}}},\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)
