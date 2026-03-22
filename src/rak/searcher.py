from __future__ import annotations

from dataclasses import dataclass

from rak.bm25 import BM25Index
from rak.embedder import Embedder
from rak.store import VectorStore


@dataclass
class SearchResult:
    doc_id: str
    score: float
    title: str = ""
    source: str = ""
    snippet: str = ""


def rrf_fuse(
    ranked_lists: list[list[dict]],
    limit: int = 10,
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    titles: dict[str, str] = {}
    snippets: dict[str, str] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if "metadata" in item and "title" in item["metadata"]:
                titles[doc_id] = item["metadata"]["title"]
            elif "title" in item:
                titles[doc_id] = item["title"]
            if doc_id not in snippets and item.get("document"):
                snippets[doc_id] = item["document"]
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]
    return [
        SearchResult(
            doc_id=doc_id, score=scores[doc_id], title=titles.get(doc_id, ""),
            source="fused", snippet=snippets.get(doc_id, ""),
        )
        for doc_id in sorted_ids
    ]


def build_where_filter(
    collection: str | None = None,
    tags: list[str] | None = None,
) -> dict | None:
    filters = []
    if collection:
        filters.append({"collections": {"$contains": collection}})
    if tags:
        if len(tags) == 1:
            filters.append({"tags": {"$contains": tags[0]}})
        else:
            filters.append({"$or": [{"tags": {"$contains": t}} for t in tags]})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _deduplicate_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate chunk results back to parent papers, keeping the best score and snippet."""
    seen: dict[str, SearchResult] = {}
    for r in results:
        parent_key = r.doc_id.split("_chunk_")[0] if "_chunk_" in r.doc_id else r.doc_id
        if parent_key not in seen or r.score > seen[parent_key].score:
            seen[parent_key] = SearchResult(
                doc_id=parent_key, score=r.score, title=r.title, source=r.source,
                snippet=r.snippet,
            )
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)


class Searcher:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_index: BM25Index) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25_index

    def vector_search(
        self, query: str, limit: int = 10,
        collection: str | None = None, tags: list[str] | None = None,
    ) -> list[SearchResult]:
        embedding = self._embedder.embed(query)
        where = build_where_filter(collection, tags)
        # Fetch extra results to account for chunk deduplication
        results = self._vector_store.search(embedding, limit=limit * 3, where=where)
        raw = [
            SearchResult(
                doc_id=r["id"], score=r["score"],
                title=r.get("metadata", {}).get("title", ""),
                source="vector",
                snippet=r.get("document", ""),
            )
            for r in results
        ]
        return _deduplicate_chunks(raw)[:limit]

    def hybrid_search(
        self, query: str, limit: int = 10,
        collection: str | None = None, tags: list[str] | None = None,
    ) -> list[SearchResult]:
        embedding = self._embedder.embed(query)
        where = build_where_filter(collection, tags)
        vector_results = self._vector_store.search(embedding, limit=limit * 3, where=where)
        bm25_results = self._bm25.search(query, limit=limit * 2)
        fused = rrf_fuse([vector_results, bm25_results], limit=limit * 3)
        return _deduplicate_chunks(fused)[:limit]
