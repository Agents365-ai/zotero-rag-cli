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
