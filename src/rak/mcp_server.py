"""MCP server for rak — exposes search and status tools for LM Studio, Cursor, etc."""
from __future__ import annotations

import atexit
import json

from mcp.server.fastmcp import FastMCP

from rak.config import RakConfig

mcp = FastMCP("rak")

_cached_searcher: tuple | None = None


def _cleanup() -> None:
    """Close cached BM25 SQLite connection on shutdown."""
    global _cached_searcher
    if _cached_searcher is not None:
        _, _, bm25 = _cached_searcher
        bm25.close()
        _cached_searcher = None


atexit.register(_cleanup)


def _get_config() -> RakConfig:
    return RakConfig()


def _init_searcher(config: RakConfig):
    global _cached_searcher
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
