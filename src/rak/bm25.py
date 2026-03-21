from __future__ import annotations

import sqlite3
from pathlib import Path


class BM25Index:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts "
            "USING fts5(doc_id UNINDEXED, content)"
        )
        self._conn.commit()

    def add(self, doc_id: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO papers_fts (doc_id, content) VALUES (?, ?)",
            (doc_id, content),
        )
        self._conn.commit()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        safe_query = " ".join(f'"{token}"' for token in query.split() if token.strip())
        if not safe_query:
            return []
        try:
            rows = self._conn.execute(
                "SELECT doc_id, rank FROM papers_fts "
                "WHERE papers_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"id": row[0], "score": -row[1]} for row in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM papers_fts").fetchone()
        return row[0]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM papers_fts")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
