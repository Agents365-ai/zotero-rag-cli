import pytest

from rak.searcher import rrf_fuse, SearchResult


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
