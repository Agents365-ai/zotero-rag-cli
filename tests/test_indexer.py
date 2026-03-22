import json
import pytest

from rak.indexer import parse_zot_items, build_document_text


def test_parse_zot_items():
    raw = json.dumps([
        {
            "key": "ABC123",
            "title": "Single Cell Analysis",
            "creators": [{"first_name": "John", "last_name": "Doe", "creator_type": "author"}],
            "abstract": "We present a method...",
            "date": "2024",
            "tags": ["scRNA-seq", "methods"],
            "item_type": "journalArticle",
        }
    ])
    items = parse_zot_items(raw)
    assert len(items) == 1
    assert items[0]["key"] == "ABC123"
    assert items[0]["title"] == "Single Cell Analysis"


def test_build_document_text():
    item = {
        "key": "ABC123",
        "title": "Single Cell Analysis",
        "creators": [{"first_name": "John", "last_name": "Doe", "creator_type": "author"}],
        "abstract": "We present a method for single cell RNA-seq.",
        "tags": ["scRNA-seq", "methods"],
    }
    text = build_document_text(item)
    assert "Single Cell Analysis" in text
    assert "John Doe" in text
    assert "single cell RNA-seq" in text
    assert "scRNA-seq" in text


def test_build_document_text_missing_fields():
    item = {"key": "X1", "title": "Minimal", "creators": [], "abstract": None, "tags": []}
    text = build_document_text(item)
    assert "Minimal" in text


def test_parse_zot_items_empty():
    items = parse_zot_items("[]")
    assert items == []


from rak.indexer import diff_items
from rak.registry import compute_hash


def _make_item(key, title, abstract=""):
    return {"key": key, "title": title, "creators": [], "abstract": abstract, "tags": []}


def test_diff_items_all_new():
    items = [_make_item("A1", "Paper One"), _make_item("A2", "Paper Two")]
    to_add, to_update, to_remove = diff_items(items, {})
    assert len(to_add) == 2
    assert len(to_update) == 0
    assert len(to_remove) == 0


def test_diff_items_unchanged():
    items = [_make_item("A1", "Paper One")]
    text = build_document_text(items[0])
    registry = {"A1": compute_hash(text)}
    to_add, to_update, to_remove = diff_items(items, registry)
    assert len(to_add) == 0
    assert len(to_update) == 0
    assert len(to_remove) == 0


def test_diff_items_updated():
    items = [_make_item("A1", "Paper One Updated")]
    registry = {"A1": "old_hash_value"}
    to_add, to_update, to_remove = diff_items(items, registry)
    assert len(to_add) == 0
    assert len(to_update) == 1
    assert to_update[0]["key"] == "A1"
    assert len(to_remove) == 0


def test_diff_items_removed():
    items = [_make_item("A1", "Paper One")]
    text = build_document_text(items[0])
    registry = {"A1": compute_hash(text), "B1": "some_hash"}
    to_add, to_update, to_remove = diff_items(items, registry)
    assert len(to_add) == 0
    assert len(to_update) == 0
    assert to_remove == ["B1"]


def test_build_document_text_with_collections():
    item = {
        "key": "C1",
        "title": "Paper",
        "creators": [],
        "abstract": None,
        "tags": ["RNA"],
        "collections": ["My Collection"],
    }
    text = build_document_text(item)
    assert "Paper" in text
    assert "RNA" in text
