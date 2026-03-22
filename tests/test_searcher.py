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


def test_deduplicate_chunks_no_chunks():
    results = [
        SearchResult(doc_id="A", score=0.9, title="Paper", source="vector"),
        SearchResult(doc_id="B", score=0.8, title="Paper 2", source="vector"),
    ]
    deduped = _deduplicate_chunks(results)
    assert len(deduped) == 2
    assert deduped[0].doc_id == "A"
