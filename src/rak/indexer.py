from __future__ import annotations

import json
import shutil
import subprocess

from pathlib import Path

from rak.bm25 import BM25Index
from rak.embedder import Embedder
from rak.errors import EmptyLibraryError, ZotNotFoundError
from rak.pdf import chunk_text, extract_pdf_text, find_pdf
from rak.registry import compute_hash
from rak.store import VectorStore


def parse_zot_items(raw_json: str) -> list[dict]:
    return json.loads(raw_json)


def build_document_text(item: dict, pdf_text: str = "") -> str:
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
    if pdf_text:
        parts.append(pdf_text)
    return "\n".join(parts)


def diff_items(
    items: list[dict], registry: dict[str, str],
    storage_dir: Path | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    to_add = []
    to_update = []
    fetched_keys = set()
    for item in items:
        key = item.get("key", "")
        if not key:
            continue
        fetched_keys.add(key)
        pdf_text = ""
        if storage_dir:
            pdf_path = find_pdf(storage_dir, key)
            if pdf_path:
                pdf_text = extract_pdf_text(pdf_path)
        text = build_document_text(item, pdf_text=pdf_text)
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
    storage_dir: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict[str, int] | int:
    if registry is not None:
        return _index_incremental(items, embedder, vector_store, bm25_index, on_progress, registry, storage_dir, chunk_size, chunk_overlap)
    return _index_full(items, embedder, vector_store, bm25_index, on_progress, storage_dir, chunk_size, chunk_overlap)


def _delete_chunks(vector_store: VectorStore, key: str) -> None:
    """Delete a document and any associated chunks from the vector store."""
    # Try deleting the base document
    vector_store.delete([key])
    # Delete any chunks (query by parent_key metadata)
    try:
        results = vector_store.get_by_metadata(where={"parent_key": key})
        if results["ids"]:
            vector_store.delete(results["ids"])
    except Exception:
        pass


def _build_metadata(item: dict) -> dict:
    collections = item.get("collections", [])
    tags = item.get("tags", [])
    metadata = {
        "title": item.get("title", ""),
        "date": item.get("date", ""),
        "item_type": item.get("item_type", ""),
    }
    if collections:
        metadata["collections"] = collections
    if tags:
        metadata["tags"] = tags
    return metadata


def _embed_and_store_batch(
    batch_ids: list[str],
    batch_texts: list[str],
    batch_metadatas: list[dict],
    embedder: Embedder,
    vector_store: VectorStore,
) -> None:
    """Embed a batch of texts and store them in the vector store."""
    if not batch_texts:
        return
    embeddings = embedder.embed_batch(batch_texts)
    vector_store.add(ids=batch_ids, embeddings=embeddings, documents=batch_texts, metadatas=batch_metadatas)


def _index_full(
    items: list[dict],
    embedder: Embedder,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    on_progress: callable | None = None,
    storage_dir: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    count = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metadatas: list[dict] = []
    embed_batch_size = 32

    for i, item in enumerate(items):
        key = item.get("key", "")
        if not key:
            continue
        pdf_text = ""
        if storage_dir:
            pdf_path = find_pdf(storage_dir, key)
            if pdf_path:
                pdf_text = extract_pdf_text(pdf_path)
        text = build_document_text(item, pdf_text=pdf_text)
        if not text.strip():
            continue
        metadata = _build_metadata(item)
        # Store full document for BM25
        bm25_index.add(key, text)
        # Accumulate chunks for batch embedding
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if len(chunks) <= 1:
            batch_ids.append(key)
            batch_texts.append(text)
            batch_metadatas.append(metadata)
        else:
            for ci, chunk in enumerate(chunks):
                chunk_id = f"{key}_chunk_{ci}"
                chunk_meta = {**metadata, "parent_key": key, "chunk_index": ci}
                batch_ids.append(chunk_id)
                batch_texts.append(chunk)
                batch_metadatas.append(chunk_meta)
        # Flush batch when full
        if len(batch_texts) >= embed_batch_size:
            _embed_and_store_batch(batch_ids, batch_texts, batch_metadatas, embedder, vector_store)
            batch_ids, batch_texts, batch_metadatas = [], [], []
        count += 1
        if on_progress and (i + 1) % 50 == 0:
            on_progress(i + 1, len(items))

    # Flush remaining
    _embed_and_store_batch(batch_ids, batch_texts, batch_metadatas, embedder, vector_store)
    return count


def _index_incremental(
    items: list[dict],
    embedder: Embedder,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    on_progress: callable | None = None,
    registry: dict[str, str] | None = None,
    storage_dir: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict[str, int]:
    to_add, to_update, to_remove = diff_items(items, registry, storage_dir=storage_dir)
    new_registry = dict(registry)
    added = 0
    updated = 0

    work_items = [(item, "add") for item in to_add] + [(item, "update") for item in to_update]
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metadatas: list[dict] = []
    embed_batch_size = 32

    for i, (item, action) in enumerate(work_items):
        key = item["key"]
        pdf_text = ""
        if storage_dir:
            pdf_path = find_pdf(storage_dir, key)
            if pdf_path:
                pdf_text = extract_pdf_text(pdf_path)
        text = build_document_text(item, pdf_text=pdf_text)
        if not text.strip():
            continue
        metadata = _build_metadata(item)
        if action == "update":
            bm25_index.delete(key)
            _delete_chunks(vector_store, key)
        bm25_index.add(key, text)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if len(chunks) <= 1:
            batch_ids.append(key)
            batch_texts.append(text)
            batch_metadatas.append(metadata)
        else:
            for ci, chunk in enumerate(chunks):
                chunk_id = f"{key}_chunk_{ci}"
                chunk_meta = {**metadata, "parent_key": key, "chunk_index": ci}
                batch_ids.append(chunk_id)
                batch_texts.append(chunk)
                batch_metadatas.append(chunk_meta)
        if len(batch_texts) >= embed_batch_size:
            _embed_and_store_batch(batch_ids, batch_texts, batch_metadatas, embedder, vector_store)
            batch_ids, batch_texts, batch_metadatas = [], [], []
        new_registry[key] = compute_hash(text)
        if action == "add":
            added += 1
        else:
            updated += 1
        if on_progress and (i + 1) % 50 == 0:
            on_progress(i + 1, len(work_items))

    _embed_and_store_batch(batch_ids, batch_texts, batch_metadatas, embedder, vector_store)

    for key in to_remove:
        _delete_chunks(vector_store, key)
        bm25_index.delete(key)
        del new_registry[key]

    return {
        "added": added,
        "updated": updated,
        "removed": len(to_remove),
        "unchanged": len(items) - added - updated,
        "registry": new_registry,
    }
