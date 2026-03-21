from __future__ import annotations

import json
import shutil
import subprocess

from rak.bm25 import BM25Index
from rak.embedder import Embedder
from rak.errors import EmptyLibraryError, ZotNotFoundError
from rak.store import VectorStore


def parse_zot_items(raw_json: str) -> list[dict]:
    return json.loads(raw_json)


def build_document_text(item: dict) -> str:
    parts = [item.get("title", "")]
    creators = item.get("creators", [])
    if creators:
        author_names = []
        for c in creators:
            name_parts = [c.get("first_name", ""), c.get("last_name", "")]
            author_names.append(" ".join(p for p in name_parts if p))
        parts.append("Authors: " + ", ".join(author_names))
    abstract = item.get("abstract")
    if abstract:
        parts.append(abstract)
    tags = item.get("tags", [])
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    return "\n".join(parts)


def fetch_zot_items(zot_command: str = "zot", limit: int = 5000) -> list[dict]:
    if not shutil.which(zot_command):
        raise ZotNotFoundError(zot_command)
    result = subprocess.run(
        [zot_command, "--json", "--limit", str(limit), "list"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"zot command failed: {result.stderr}")
    items = parse_zot_items(result.stdout)
    if not items:
        raise EmptyLibraryError()
    return items


def index_items(
    items: list[dict],
    embedder: Embedder,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    on_progress: callable | None = None,
) -> int:
    count = 0
    for i, item in enumerate(items):
        key = item.get("key", "")
        if not key:
            continue
        text = build_document_text(item)
        if not text.strip():
            continue
        embedding = embedder.embed(text)
        metadata = {
            "title": item.get("title", ""),
            "date": item.get("date", ""),
            "item_type": item.get("item_type", ""),
        }
        vector_store.add(ids=[key], embeddings=[embedding], documents=[text], metadatas=[metadata])
        bm25_index.add(key, text)
        count += 1
        if on_progress and (i + 1) % 50 == 0:
            on_progress(i + 1, len(items))
    return count
