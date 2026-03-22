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


def rrf_fuse(
    ranked_lists: list[list[dict]],
    limit: int = 10,
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    titles: dict[str, str] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if "metadata" in item and "title" in item["metadata"]:
                titles[doc_id] = item["metadata"]["title"]
            elif "title" in item:
                titles[doc_id] = item["title"]
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]
    return [
        SearchResult(doc_id=doc_id, score=scores[doc_id], title=titles.get(doc_id, ""), source="fused")
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
        results = self._vector_store.search(embedding, limit=limit, where=where)
        return [
            SearchResult(doc_id=r["id"], score=r["score"], title=r.get("metadata", {}).get("title", ""), source="vector")
            for r in results
        ]

    def hybrid_search(
        self, query: str, limit: int = 10,
        collection: str | None = None, tags: list[str] | None = None,
    ) -> list[SearchResult]:
        embedding = self._embedder.embed(query)
        where = build_where_filter(collection, tags)
        vector_results = self._vector_store.search(embedding, limit=limit * 2, where=where)
        bm25_results = self._bm25.search(query, limit=limit * 2)
        return rrf_fuse([vector_results, bm25_results], limit=limit)
