import pytest

from rak.bm25 import BM25Index


@pytest.fixture
def bm25(tmp_path):
    return BM25Index(tmp_path / "fts.sqlite")


def test_add_and_search(bm25):
    bm25.add("A1", "single cell RNA sequencing analysis")
    bm25.add("A2", "deep learning for image classification")
    results = bm25.search("single cell RNA", limit=5)
    assert len(results) >= 1
    assert results[0]["id"] == "A1"


def test_search_ranking(bm25):
    bm25.add("A1", "RNA RNA RNA sequencing single cell")
    bm25.add("A2", "RNA processing")
    results = bm25.search("RNA", limit=2)
    assert results[0]["id"] == "A1"


def test_search_no_results(bm25):
    bm25.add("A1", "deep learning")
    results = bm25.search("quantum physics", limit=5)
    assert len(results) == 0


def test_count(bm25):
    assert bm25.count() == 0
    bm25.add("A1", "test document")
    assert bm25.count() == 1


def test_clear(bm25):
    bm25.add("A1", "test")
    bm25.clear()
    assert bm25.count() == 0


def test_search_returns_score(bm25):
    bm25.add("A1", "transformer attention mechanism")
    results = bm25.search("transformer", limit=1)
    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_delete(bm25):
    bm25.add("A1", "paper about RNA")
    bm25.add("A2", "paper about DNA")
    assert bm25.count() == 2
    bm25.delete("A1")
    assert bm25.count() == 1


def test_search_query_with_quotes(bm25):
    bm25.add("A1", "he said hello world")
    results = bm25.search('he said "hello"', limit=5)
    assert isinstance(results, list)


def test_search_with_snippet(bm25):
    bm25.add("A1", "single cell RNA sequencing analysis method")
    bm25.add("A2", "deep learning for image classification")
    results = bm25.search_with_snippet("single cell RNA", limit=5)
    assert len(results) >= 1
    assert results[0]["id"] == "A1"
    assert "snippet" in results[0]
    assert results[0]["snippet"]  # non-empty


def test_search_with_snippet_no_results(bm25):
    bm25.add("A1", "deep learning")
    results = bm25.search_with_snippet("quantum physics", limit=5)
    assert results == []


def test_search_with_snippet_has_title_key(bm25):
    bm25.add("A1", "transformer attention mechanism")
    results = bm25.search_with_snippet("transformer", limit=1)
    assert "title" in results[0]
