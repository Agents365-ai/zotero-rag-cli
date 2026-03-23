import pytest

from rak.searcher import _deduplicate_chunks, rrf_fuse, SearchResult


def test_rrf_fuse_single_source():
    vector_results = [
        {"id": "A", "score": 0.9},
        {"id": "B", "score": 0.7},
    ]
    fused = rrf_fuse([vector_results], limit=2)
    assert fused[0].doc_id == "A"
    assert fused[1].doc_id == "B"


def test_rrf_fuse_two_sources():
    vector_results = [
        {"id": "A", "score": 0.9},
        {"id": "B", "score": 0.5},
    ]
    bm25_results = [
        {"id": "B", "score": 5.0},
        {"id": "C", "score": 3.0},
    ]
    fused = rrf_fuse([vector_results, bm25_results], limit=3)
    ids = [r.doc_id for r in fused]
    assert "B" in ids[:2]


def test_rrf_fuse_empty():
    fused = rrf_fuse([], limit=5)
    assert fused == []


def test_rrf_fuse_respects_limit():
    results = [{"id": f"X{i}", "score": float(i)} for i in range(20)]
    fused = rrf_fuse([results], limit=5)
    assert len(fused) == 5


def test_search_result_dataclass():
    r = SearchResult(doc_id="A1", score=0.85, title="Test Paper", source="vector")
    assert r.doc_id == "A1"
    assert r.score == 0.85


from rak.searcher import build_where_filter


def test_build_where_filter_collection_only():
    f = build_where_filter(collection="My Papers", tags=None)
    assert f == {"collections": {"$contains": "My Papers"}}


def test_build_where_filter_single_tag():
    f = build_where_filter(collection=None, tags=["RNA"])
    assert f == {"tags": {"$contains": "RNA"}}


def test_build_where_filter_multiple_tags():
    f = build_where_filter(collection=None, tags=["RNA", "DNA"])
    assert f == {"$or": [{"tags": {"$contains": "RNA"}}, {"tags": {"$contains": "DNA"}}]}


def test_build_where_filter_collection_and_tags():
    f = build_where_filter(collection="Bio", tags=["RNA"])
    assert f == {"$and": [{"collections": {"$contains": "Bio"}}, {"tags": {"$contains": "RNA"}}]}


def test_build_where_filter_none():
    f = build_where_filter(collection=None, tags=None)
    assert f is None


def test_deduplicate_chunks_merges_same_parent():
    results = [
        SearchResult(doc_id="ABC_chunk_0", score=0.9, title="Paper A", source="vector", snippet="best chunk"),
        SearchResult(doc_id="ABC_chunk_1", score=0.8, title="Paper A", source="vector", snippet="other chunk"),
        SearchResult(doc_id="DEF", score=0.7, title="Paper B", source="vector", snippet="only chunk"),
    ]
    deduped = _deduplicate_chunks(results)
    ids = [r.doc_id for r in deduped]
    assert ids == ["ABC", "DEF"]
    assert deduped[0].score == 0.9
    assert deduped[0].snippet == "best chunk"


def test_rrf_fuse_preserves_snippets():
    vector_results = [
        {"id": "A", "score": 0.9, "document": "snippet text A"},
        {"id": "B", "score": 0.7, "document": "snippet text B"},
    ]
    bm25_results = [
        {"id": "B", "score": 5.0},
        {"id": "C", "score": 3.0},
    ]
    fused = rrf_fuse([vector_results, bm25_results], limit=3)
    snippets = {r.doc_id: r.snippet for r in fused}
    assert snippets["A"] == "snippet text A"
    assert snippets["B"] == "snippet text B"
    assert snippets["C"] == ""


def test_bm25_search_standalone():
    """Test BM25-only search via Searcher (no embedder/vector store needed)."""
    from unittest.mock import MagicMock
    bm25 = MagicMock()
    bm25.search_with_snippet.return_value = [
        {"id": "A1", "score": 5.0, "snippet": "RNA analysis...", "title": ""},
        {"id": "B1", "score": 3.0, "snippet": "cell method...", "title": ""},
    ]
    from rak.searcher import Searcher
    searcher = Searcher(None, None, bm25)
    results = searcher.bm25_search("RNA cell", limit=10)
    assert len(results) == 2
    assert results[0].doc_id == "A1"
    assert results[0].source == "bm25"
    assert results[0].snippet == "RNA analysis..."


def test_bm25_search_deduplicates_chunks():
    from unittest.mock import MagicMock
    bm25 = MagicMock()
    bm25.search_with_snippet.return_value = [
        {"id": "A1_chunk_0", "score": 5.0, "snippet": "best chunk", "title": ""},
        {"id": "A1_chunk_1", "score": 3.0, "snippet": "other chunk", "title": ""},
        {"id": "B1", "score": 2.0, "snippet": "another paper", "title": ""},
    ]
    from rak.searcher import Searcher
    searcher = Searcher(None, None, bm25)
    results = searcher.bm25_search("test", limit=10)
    ids = [r.doc_id for r in results]
    assert ids == ["A1", "B1"]


def test_deduplicate_chunks_no_chunks():
    results = [
        SearchResult(doc_id="A", score=0.9, title="Paper", source="vector"),
        SearchResult(doc_id="B", score=0.8, title="Paper 2", source="vector"),
    ]
    deduped = _deduplicate_chunks(results)
    assert len(deduped) == 2
    assert deduped[0].doc_id == "A"


def test_similar_search_excludes_source():
    """similar_search returns similar papers excluding the source paper."""
    from unittest.mock import MagicMock
    from rak.searcher import Searcher

    vector_store = MagicMock()
    vector_store.get_embedding.return_value = [0.1, 0.2, 0.3]
    vector_store.search.return_value = [
        {"id": "SRC", "score": 1.0, "document": "source paper", "metadata": {"title": "Source"}},
        {"id": "A", "score": 0.9, "document": "similar paper A", "metadata": {"title": "Paper A"}},
        {"id": "B", "score": 0.8, "document": "similar paper B", "metadata": {"title": "Paper B"}},
    ]
    bm25 = MagicMock()
    searcher = Searcher(MagicMock(), vector_store, bm25)
    results = searcher.similar_search("SRC", limit=5)
    ids = [r.doc_id for r in results]
    assert "SRC" not in ids
    assert "A" in ids
    assert "B" in ids
    assert results[0].source == "similar"


def test_similar_search_uses_chunk_embedding():
    """similar_search falls back to chunk_0 embedding when key not found."""
    from unittest.mock import MagicMock
    from rak.searcher import Searcher

    vector_store = MagicMock()
    vector_store.get_embedding.side_effect = lambda key: None if key == "SRC" else [0.1, 0.2]
    vector_store.search.return_value = [
        {"id": "SRC_chunk_0", "score": 1.0, "document": "chunk", "metadata": {"title": "Source"}},
        {"id": "A", "score": 0.9, "document": "other", "metadata": {"title": "Paper A"}},
    ]
    bm25 = MagicMock()
    searcher = Searcher(MagicMock(), vector_store, bm25)
    results = searcher.similar_search("SRC", limit=5)
    vector_store.get_embedding.assert_any_call("SRC_chunk_0")
    ids = [r.doc_id for r in results]
    assert "SRC" not in ids


def test_similar_search_key_not_found():
    """similar_search returns empty list when key doesn't exist."""
    from unittest.mock import MagicMock
    from rak.searcher import Searcher

    vector_store = MagicMock()
    vector_store.get_embedding.return_value = None
    bm25 = MagicMock()
    searcher = Searcher(MagicMock(), vector_store, bm25)
    results = searcher.similar_search("NONEXIST", limit=5)
    assert results == []


def test_similar_search_no_vector_store():
    """similar_search returns empty when no vector store (BM25-only mode)."""
    from unittest.mock import MagicMock
    from rak.searcher import Searcher

    bm25 = MagicMock()
    searcher = Searcher(None, None, bm25)
    results = searcher.similar_search("SRC", limit=5)
    assert results == []


def test_similar_search_with_filters():
    """similar_search passes where filter to vector_store.search when collection/tags given."""
    from unittest.mock import MagicMock, call
    from rak.searcher import Searcher, build_where_filter

    vector_store = MagicMock()
    vector_store.get_embedding.return_value = [0.1, 0.2, 0.3]
    vector_store.search.return_value = [
        {"id": "A", "score": 0.9, "document": "paper A", "metadata": {"title": "Paper A"}},
    ]
    bm25 = MagicMock()
    searcher = Searcher(MagicMock(), vector_store, bm25)

    results = searcher.similar_search("SRC", limit=5, collection="My Collection", tags=["RNA"])

    expected_where = build_where_filter(collection="My Collection", tags=["RNA"])
    vector_store.search.assert_called_once_with(
        [0.1, 0.2, 0.3], limit=18, where=expected_where
    )
    assert results[0].doc_id == "A"
