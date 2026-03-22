from __future__ import annotations

import json
import shutil
import subprocess

from rak.bm25 import BM25Index
from rak.embedder import Embedder
from rak.errors import EmptyLibraryError, ZotNotFoundError
from rak.registry import compute_hash
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


def diff_items(
    items: list[dict], registry: dict[str, str]
) -> tuple[list[dict], list[dict], list[str]]:
    to_add = []
    to_update = []
    fetched_keys = set()
    for item in items:
        key = item.get("key", "")
        if not key:
            continue
        fetched_keys.add(key)
        text = build_document_text(item)
        if not text.strip():
            continue
        content_hash = compute_hash(text)
        if key not in registry:
            to_add.append(item)
        elif registry[key] != content_hash:
            to_update.append(item)
    to_remove = [k for k in registry if k not in fetched_keys]
    return to_add, to_update, to_remove


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
    registry: dict[str, str] | None = None,
) -> dict[str, int] | int:
    if registry is not None:
        return _index_incremental(items, embedder, vector_store, bm25_index, on_progress, registry)
    return _index_full(items, embedder, vector_store, bm25_index, on_progress)


def _index_full(
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


def _index_incremental(
    items: list[dict],
    embedder: Embedder,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    on_progress: callable | None = None,
    registry: dict[str, str] = None,
) -> dict[str, int]:
    to_add, to_update, to_remove = diff_items(items, registry)
    new_registry = dict(registry)
    added = 0
    updated = 0

    work_items = [(item, "add") for item in to_add] + [(item, "update") for item in to_update]
    for i, (item, action) in enumerate(work_items):
        key = item["key"]
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
        if action == "update":
            bm25_index.delete(key)
        bm25_index.add(key, text)
        new_registry[key] = compute_hash(text)
        if action == "add":
            added += 1
        else:
            updated += 1
        if on_progress and (i + 1) % 50 == 0:
            on_progress(i + 1, len(work_items))

    for key in to_remove:
        vector_store.delete([key])
        bm25_index.delete(key)
        del new_registry[key]

    return {
        "added": added,
        "updated": updated,
        "removed": len(to_remove),
        "unchanged": len(items) - added - updated,
        "registry": new_registry,
    }
