"""MCP server for rak — exposes search and status tools for LM Studio, Cursor, etc."""
from __future__ import annotations

import atexit
import json
import threading

from mcp.server.fastmcp import FastMCP

from rak.config import RakConfig

mcp = FastMCP("rak")

_cached_searcher: tuple | None = None
_searcher_lock = threading.Lock()


def _cleanup() -> None:
    """Close cached BM25 SQLite connection on shutdown."""
    global _cached_searcher
    with _searcher_lock:
        if _cached_searcher is not None:
            _, _, bm25 = _cached_searcher
            bm25.close()
            _cached_searcher = None


atexit.register(_cleanup)


def _get_config() -> RakConfig:
    return RakConfig()


def _init_searcher(config: RakConfig):
    global _cached_searcher
    with _searcher_lock:
        if _cached_searcher is not None:
            # Validate cached BM25 connection is still usable
            _, _, bm25 = _cached_searcher
            try:
                if not config.fts_db_path.exists():
                    raise FileNotFoundError
                bm25.search("", limit=1)  # lightweight probe
            except Exception:
                bm25.close()
                _cached_searcher = None

        if _cached_searcher is not None:
            return _cached_searcher

        from rak.bm25 import BM25Index
        from rak.embedder import Embedder
        from rak.searcher import Searcher
        from rak.store import VectorStore

        embedder = Embedder(config.model_name, provider=config.embedding_provider, base_url=config.embedding_base_url, api_key=config.embedding_api_key)
        vector_store = VectorStore(config.chroma_dir, embedder.dimension)
        bm25 = BM25Index(config.fts_db_path)
        searcher = Searcher(embedder, vector_store, bm25)
        _cached_searcher = (searcher, vector_store, bm25)
        return _cached_searcher


@mcp.tool()
def search_papers(query: str, limit: int = 10, hybrid: bool = False,
                  collection: str | None = None, tags: list[str] | None = None) -> str:
    """Search your Zotero library using semantic or hybrid search.

    Args:
        query: Search query (natural language)
        limit: Maximum number of results (default 10)
        hybrid: Use hybrid search combining vector + BM25 keyword search
        collection: Filter by Zotero collection name
        tags: Filter by tags (OR logic)

    Returns:
        JSON array of search results with key, title, score, and source.
    """
    config = _get_config()
    searcher, _, bm25 = _init_searcher(config)

    tag_list = tags if tags else None
    if hybrid:
        results = searcher.hybrid_search(query, limit=limit, collection=collection, tags=tag_list)
    else:
        results = searcher.vector_search(query, limit=limit, collection=collection, tags=tag_list)

    output = []
    for r in results:
        item = {"key": r.doc_id, "title": r.title, "score": round(r.score, 4), "source": r.source}
        if r.snippet:
            item["snippet"] = r.snippet
        output.append(item)

    return json.dumps(output, indent=2, ensure_ascii=False)


@mcp.tool()
def search_papers_bm25(query: str, limit: int = 10) -> str:
    """Search your Zotero library using pure BM25 keyword search (no embedding model needed).

    Args:
        query: Search query (keywords)
        limit: Maximum number of results (default 10)

    Returns:
        JSON array of search results with key, title, score, snippet, and source.
    """
    from rak.bm25 import BM25Index
    from rak.searcher import Searcher

    config = _get_config()
    with BM25Index(config.fts_db_path) as bm25:
        searcher = Searcher(None, None, bm25)
        results = searcher.bm25_search(query, limit=limit)

    output = []
    for r in results:
        item = {"key": r.doc_id, "title": r.title, "score": round(r.score, 4), "source": r.source}
        if r.snippet:
            item["snippet"] = r.snippet
        output.append(item)

    return json.dumps(output, indent=2, ensure_ascii=False)


def _resolve_key_mcp(key_or_title: str, vector_store, bm25) -> tuple[str | None, list[dict] | None]:
    """Resolve a key or title to a Zotero key for MCP (non-interactive).

    Returns (resolved_key, candidates_or_none). If multiple matches,
    returns (None, candidates) so the caller can return them to the AI.
    """
    import re
    if re.match(r'^[A-Za-z0-9]{4,12}$', key_or_title):
        if vector_store.has(key_or_title) or vector_store.has(f"{key_or_title}_chunk_0"):
            return key_or_title, None

    results = bm25.search(key_or_title, limit=10)
    if not results:
        return None, None

    parent_ids = list(dict.fromkeys(
        doc_id.split("_chunk_")[0] if "_chunk_" in doc_id else doc_id
        for doc_id in [r["id"] for r in results]
    ))
    doc_data = vector_store.get(ids=parent_ids, include=["metadatas"])
    candidates = [{"key": doc_id, "title": meta.get("title", doc_id)}
                  for doc_id, meta in zip(doc_data["ids"], doc_data["metadatas"])]

    if len(candidates) == 1:
        return candidates[0]["key"], None
    return None, candidates


@mcp.tool()
def similar_papers(key_or_title: str, limit: int = 10,
                   collection: str | None = None, tags: list[str] | None = None) -> str:
    """Find papers similar to a given one by its Zotero key or title.

    Args:
        key_or_title: Zotero item key (e.g., ABC12345) or title search query (e.g., "attention is all you need")
        limit: Maximum number of similar papers to return (default 10)
        collection: Filter by Zotero collection name
        tags: Filter by tags (OR logic)

    Returns:
        JSON array of similar papers, or candidate list if multiple title matches found.
    """
    config = _get_config()
    searcher, vector_store, bm25 = _init_searcher(config)

    key, candidates = _resolve_key_mcp(key_or_title, vector_store, bm25)
    if key is None and candidates:
        return json.dumps({"status": "multiple_matches", "query": key_or_title, "candidates": candidates}, indent=2)
    if key is None:
        return json.dumps({"status": "not_found", "query": key_or_title})

    tag_list = tags if tags else None
    results = searcher.similar_search(key, limit=limit, collection=collection, tags=tag_list)

    output = []
    for r in results:
        item = {"key": r.doc_id, "title": r.title, "score": round(r.score, 4), "source": r.source}
        if r.snippet:
            item["snippet"] = r.snippet
        output.append(item)

    return json.dumps(output, indent=2, ensure_ascii=False)


@mcp.tool()
def ask_papers(question: str, limit: int = 5, hybrid: bool = False,
               collection: str | None = None, tags: list[str] | None = None) -> str:
    """Ask a question and get an LLM-generated answer based on your indexed papers.

    Args:
        question: The question to answer
        limit: Number of papers to use as context (default 5)
        hybrid: Use hybrid search for context retrieval
        collection: Filter by Zotero collection name
        tags: Filter by tags (OR logic)

    Returns:
        JSON object with answer text and source papers used.
    """
    from rak.llm import LLMClient

    config = _get_config()
    searcher, _, _ = _init_searcher(config)

    tag_list = tags if tags else None
    if hybrid:
        results = searcher.hybrid_search(question, limit=limit, collection=collection, tags=tag_list)
    else:
        results = searcher.vector_search(question, limit=limit, collection=collection, tags=tag_list)

    if not results:
        return json.dumps({"answer": "No relevant papers found.", "sources": []})

    context = [{"key": r.doc_id, "title": r.title, "text": r.snippet, "score": r.score} for r in results]

    llm = LLMClient(base_url=config.llm_base_url, model=config.llm_model, api_key=config.llm_api_key)
    answer = llm.ask(question, context)

    return json.dumps({
        "answer": answer,
        "sources": [{"key": c["key"], "title": c["title"], "score": round(c["score"], 4)} for c in context],
    }, indent=2, ensure_ascii=False)


@mcp.tool()
def export_papers(query: str, limit: int = 10, format: str = "csv",
                  hybrid: bool = False, collection: str | None = None,
                  tags: list[str] | None = None) -> str:
    """Export search results as CSV or BibTeX.

    Args:
        query: Search query
        limit: Maximum number of results (default 10)
        format: Export format, either "csv" or "bibtex" (default "csv")
        hybrid: Use hybrid search
        collection: Filter by Zotero collection name
        tags: Filter by tags (OR logic)

    Returns:
        Formatted CSV or BibTeX string.
    """
    from rak.export import to_bibtex, to_csv

    config = _get_config()
    searcher, vector_store, _ = _init_searcher(config)

    tag_list = tags if tags else None
    if hybrid:
        results = searcher.hybrid_search(query, limit=limit, collection=collection, tags=tag_list)
    else:
        results = searcher.vector_search(query, limit=limit, collection=collection, tags=tag_list)

    if not results:
        return "No results found."

    # Batch fetch metadata
    meta_map: dict[str, dict] = {}
    if vector_store is not None:
        all_ids = [r.doc_id for r in results]
        doc_data = vector_store.get(ids=all_ids, include=["metadatas"])
        for doc_id, meta in zip(doc_data["ids"], doc_data["metadatas"]):
            meta_map[doc_id] = meta

    export_rows = []
    for r in results:
        meta = meta_map.get(r.doc_id, {})
        export_rows.append({
            "key": r.doc_id, "title": r.title, "score": r.score, "source": r.source,
            "date": meta.get("date", ""), "authors": meta.get("authors", ""),
            "item_type": meta.get("item_type", ""),
        })

    if format == "bibtex":
        return to_bibtex(export_rows)
    return to_csv(export_rows)


@mcp.tool()
def show_config() -> str:
    """Show current rak configuration values.

    Returns:
        JSON object with all configuration key-value pairs.
    """
    config = _get_config()
    return json.dumps({
        "model_name": config.model_name,
        "zot_command": config.zot_command,
        "llm_base_url": config.llm_base_url,
        "llm_model": config.llm_model,
        "embedding_provider": config.embedding_provider,
        "embedding_base_url": config.embedding_base_url if config.embedding_provider == "api" else None,
        "pdf_provider": config.pdf_provider,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "data_dir": str(config.data_dir),
        "zotero_storage_dir": str(config.zotero_storage_dir) if config.zotero_storage_dir else None,
    }, indent=2)


@mcp.tool()
def index_status() -> str:
    """Show the current index status including item count, model, and last indexed time.

    Returns:
        JSON object with index metadata or a message if no index exists.
    """
    from rak.metadata import load_metadata

    config = _get_config()
    meta = load_metadata(config.data_dir)

    if meta is None:
        return json.dumps({"status": "No index found. Run 'rak index' first."})

    return json.dumps({
        "item_count": meta.item_count,
        "model_name": meta.model_name,
        "last_indexed": meta.last_indexed,
        "data_directory": str(config.data_dir),
    }, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
