"""Tests for rak.formatter — format_results, format_index_stats,
format_incremental_stats, format_ask_result."""

import json

from rak.formatter import (
    format_ask_result,
    format_incremental_stats,
    format_index_stats,
    format_results,
)
from rak.searcher import SearchResult


# --- format_results ---


def test_format_results_json():
    results = [
        SearchResult(doc_id="A1", score=0.9512, title="Paper One", source="vector", snippet="chunk text"),
        SearchResult(doc_id="A2", score=0.7, title="Paper Two", source="bm25"),
    ]
    out = format_results(results, output_json=True)
    data = json.loads(out)
    assert len(data) == 2
    assert data[0]["key"] == "A1"
    assert data[0]["title"] == "Paper One"
    assert data[0]["score"] == 0.9512
    assert data[0]["source"] == "vector"
    assert data[0]["snippet"] == "chunk text"
    # No snippet for second result
    assert "snippet" not in data[1]


def test_format_results_json_empty():
    out = format_results([], output_json=True)
    assert json.loads(out) == []


def test_format_results_table():
    results = [
        SearchResult(doc_id="X1", score=0.95, title="RNA Sequencing", source="vector"),
    ]
    out = format_results(results, output_json=False)
    assert "X1" in out
    assert "RNA Sequencing" in out
    assert "0.950" in out
    assert "vector" in out


def test_format_results_table_empty():
    out = format_results([], output_json=False)
    # Rich table with no rows should still have header or be mostly empty
    assert isinstance(out, str)


# --- format_index_stats ---


def test_format_index_stats_json():
    out = format_index_stats(42, output_json=True)
    data = json.loads(out)
    assert data == {"indexed": 42}


def test_format_index_stats_text():
    out = format_index_stats(42, output_json=False)
    assert out == "Indexed 42 papers."


# --- format_incremental_stats ---


def test_format_incremental_stats_json():
    stats = {"added": 5, "updated": 2, "removed": 1, "unchanged": 10, "registry": {}, "text_cache": {}}
    out = format_incremental_stats(stats, output_json=True)
    data = json.loads(out)
    assert data["added"] == 5
    assert data["updated"] == 2
    assert data["removed"] == 1
    assert data["unchanged"] == 10
    # registry and text_cache should be excluded
    assert "registry" not in data
    assert "text_cache" not in data


def test_format_incremental_stats_text_all_changes():
    stats = {"added": 3, "updated": 2, "removed": 1, "unchanged": 10}
    out = format_incremental_stats(stats, output_json=False)
    assert "3 new" in out
    assert "2 updated" in out
    assert "1 removed" in out
    assert "10 unchanged" in out
    assert out.endswith(".")


def test_format_incremental_stats_text_only_added():
    stats = {"added": 5, "updated": 0, "removed": 0, "unchanged": 0}
    out = format_incremental_stats(stats, output_json=False)
    assert "5 new" in out
    assert "updated" not in out
    assert "removed" not in out
    assert "0 unchanged" in out


def test_format_incremental_stats_text_no_changes():
    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 50}
    out = format_incremental_stats(stats, output_json=False)
    assert "50 unchanged" in out
    assert "new" not in out
    assert "updated" not in out
    assert "removed" not in out


# --- format_ask_result ---


def test_format_ask_result_json():
    sources = [
        {"key": "A1", "title": "Paper One", "score": 0.91234},
        {"key": "A2", "title": "Paper Two", "score": 0.71111},
    ]
    out = format_ask_result("The answer is 42.", sources, output_json=True)
    data = json.loads(out)
    assert data["answer"] == "The answer is 42."
    assert len(data["sources"]) == 2
    assert data["sources"][0]["key"] == "A1"
    assert data["sources"][0]["score"] == 0.9123  # rounded to 4 decimals
    assert data["sources"][1]["score"] == 0.7111


def test_format_ask_result_text():
    sources = [
        {"key": "A1", "title": "Paper One", "score": 0.912},
        {"key": "B2", "title": "Paper Two", "score": 0.700},
    ]
    out = format_ask_result("Deep learning is great.", sources, output_json=False)
    assert "Deep learning is great." in out
    assert "Sources:" in out
    assert "1. A1 - Paper One (score: 0.912)" in out
    assert "2. B2 - Paper Two (score: 0.700)" in out


def test_format_ask_result_text_no_sources():
    out = format_ask_result("No context available.", [], output_json=False)
    assert "No context available." in out
    assert "Sources:" in out


def test_format_ask_result_json_empty_sources():
    out = format_ask_result("answer", [], output_json=True)
    data = json.loads(out)
    assert data["sources"] == []
