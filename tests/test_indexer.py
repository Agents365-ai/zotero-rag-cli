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
    to_add, to_update, to_remove, _text_cache = diff_items(items, {})
    assert len(to_add) == 2
    assert len(to_update) == 0
    assert len(to_remove) == 0


def test_diff_items_unchanged():
    items = [_make_item("A1", "Paper One")]
    text = build_document_text(items[0])
    registry = {"A1": compute_hash(text)}
    to_add, to_update, to_remove, _text_cache = diff_items(items, registry)
    assert len(to_add) == 0
    assert len(to_update) == 0
    assert len(to_remove) == 0


def test_diff_items_updated():
    items = [_make_item("A1", "Paper One Updated")]
    registry = {"A1": "old_hash_value"}
    to_add, to_update, to_remove, _text_cache = diff_items(items, registry)
    assert len(to_add) == 0
    assert len(to_update) == 1
    assert to_update[0]["key"] == "A1"
    assert len(to_remove) == 0


def test_diff_items_removed():
    items = [_make_item("A1", "Paper One")]
    text = build_document_text(items[0])
    registry = {"A1": compute_hash(text), "B1": "some_hash"}
    to_add, to_update, to_remove, _text_cache = diff_items(items, registry)
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


def test_build_document_text_with_pdf():
    item = {
        "key": "P1",
        "title": "Paper With PDF",
        "creators": [],
        "abstract": "Short abstract.",
        "tags": [],
    }
    text = build_document_text(item, pdf_text="Full text of the paper extracted from PDF.")
    assert "Paper With PDF" in text
    assert "Short abstract." in text
    assert "Full text of the paper extracted from PDF." in text


# --- Parallel extraction tests ---

from pathlib import Path
from unittest.mock import patch, MagicMock
from rak.indexer import _extract_item_text, _MAX_EXTRACT_WORKERS


def test_extract_item_text_no_storage_dir():
    item = _make_item("K1", "Title")
    text, attempts, failures = _extract_item_text(item, None, "pymupdf")
    assert text == ""
    assert attempts == 0
    assert failures == 0


def test_extract_item_text_no_key():
    item = {"title": "No Key", "creators": [], "abstract": "", "tags": []}
    text, attempts, failures = _extract_item_text(item, Path("/tmp"), "pymupdf")
    assert text == ""
    assert attempts == 0


def test_extract_item_text_with_attachments(tmp_path):
    """Verify _extract_item_text finds and extracts files."""
    key = "ITEM1"
    item_dir = tmp_path / key
    item_dir.mkdir()
    (item_dir / "paper.pdf").write_bytes(b"")  # empty PDF
    (item_dir / "notes.md").write_text("Some markdown notes", encoding="utf-8")

    with patch("rak.indexer.extract_file_text") as mock_extract:
        mock_extract.side_effect = lambda p, provider="pymupdf": (
            "Some markdown notes" if p.suffix == ".md" else ""
        )
        text, attempts, failures = _extract_item_text(
            {"key": key}, tmp_path, "pymupdf"
        )
    assert attempts == 2
    assert failures == 1  # the empty PDF returns ""
    assert "Some markdown notes" in text


def test_diff_items_parallel_extraction(tmp_path):
    """Verify diff_items uses parallel extraction with storage_dir."""
    items = [_make_item("X1", "Paper X"), _make_item("X2", "Paper Y")]

    # Create storage dirs with markdown files
    for item in items:
        d = tmp_path / item["key"]
        d.mkdir()
        (d / "doc.md").write_text(f"Content for {item['key']}", encoding="utf-8")

    to_add, to_update, to_remove, text_cache = diff_items(
        items, {}, storage_dir=tmp_path, pdf_provider="pymupdf"
    )
    assert len(to_add) == 2
    assert "X1" in text_cache
    assert "X2" in text_cache
    assert "Content for X1" in text_cache["X1"]
    assert "Content for X2" in text_cache["X2"]


def test_diff_items_parallel_failure_logging(tmp_path, caplog):
    """Verify extraction failure logging works with parallel extraction."""
    items = [_make_item("F1", "Fail Paper")]
    d = tmp_path / "F1"
    d.mkdir()
    (d / "a.pdf").write_bytes(b"")
    (d / "b.pdf").write_bytes(b"")
    (d / "c.pdf").write_bytes(b"")

    with patch("rak.indexer.extract_file_text", return_value=""):
        import logging
        with caplog.at_level(logging.WARNING):
            to_add, _, _, _ = diff_items(
                items, {}, storage_dir=tmp_path, pdf_provider="pymupdf"
            )
    # All 3 extractions fail -> >50% failure rate -> warning logged
    assert any("file extractions failed" in r.message for r in caplog.records)


def test_max_extract_workers_is_reasonable():
    """Sanity check that the worker count is bounded."""
    assert 1 <= _MAX_EXTRACT_WORKERS <= 8


# --- _build_metadata tests ---

from rak.indexer import _build_metadata, _delete_chunks


def test_build_metadata_basic():
    item = {
        "key": "A1",
        "title": "Test Paper",
        "date": "2024",
        "item_type": "journalArticle",
        "creators": [],
        "collections": [],
        "tags": [],
    }
    meta = _build_metadata(item)
    assert meta["title"] == "Test Paper"
    assert meta["date"] == "2024"
    assert meta["item_type"] == "journalArticle"
    assert "authors" not in meta
    assert "collections" not in meta
    assert "tags" not in meta


def test_build_metadata_with_authors():
    item = {
        "key": "A1",
        "title": "Paper",
        "date": "2024",
        "item_type": "journalArticle",
        "creators": [
            {"first_name": "John", "last_name": "Doe"},
            {"first_name": "Jane", "last_name": "Smith"},
        ],
        "collections": [],
        "tags": [],
    }
    meta = _build_metadata(item)
    assert meta["authors"] == "John Doe, Jane Smith"


def test_build_metadata_with_collections_and_tags():
    item = {
        "key": "A1",
        "title": "Paper",
        "date": "",
        "item_type": "",
        "creators": [],
        "collections": ["ML Papers", "Favorites"],
        "tags": ["deep-learning", "NLP"],
    }
    meta = _build_metadata(item)
    assert meta["collections"] == ["ML Papers", "Favorites"]
    assert meta["tags"] == ["deep-learning", "NLP"]


def test_build_metadata_missing_fields():
    """Missing fields should use empty defaults."""
    item = {"key": "X1"}
    meta = _build_metadata(item)
    assert meta["title"] == ""
    assert meta["date"] == ""
    assert meta["item_type"] == ""


def test_build_metadata_author_partial_name():
    """Authors with only last name should work."""
    item = {
        "key": "A1",
        "title": "Paper",
        "date": "",
        "item_type": "",
        "creators": [{"first_name": "", "last_name": "Einstein"}],
        "collections": [],
        "tags": [],
    }
    meta = _build_metadata(item)
    assert meta["authors"] == "Einstein"


# --- _delete_chunks tests ---


def test_delete_chunks_removes_base_and_chunks():
    mock_store = MagicMock()
    mock_store.get_ids_by_metadata.return_value = {"ids": ["K1_chunk_0", "K1_chunk_1"]}
    _delete_chunks(mock_store, "K1")
    mock_store.delete.assert_any_call(["K1"])
    mock_store.delete.assert_any_call(["K1_chunk_0", "K1_chunk_1"])


def test_delete_chunks_no_chunks():
    mock_store = MagicMock()
    mock_store.get_ids_by_metadata.return_value = {"ids": []}
    _delete_chunks(mock_store, "K1")
    mock_store.delete.assert_called_once_with(["K1"])


def test_delete_chunks_handles_exception():
    """If get_ids_by_metadata fails, should not raise."""
    mock_store = MagicMock()
    mock_store.get_ids_by_metadata.side_effect = Exception("chromadb error")
    _delete_chunks(mock_store, "K1")  # should not raise
    mock_store.delete.assert_called_once_with(["K1"])


# --- diff_items text_cache and edge cases ---


def test_diff_items_returns_text_cache():
    items = [_make_item("A1", "Paper One", "An abstract")]
    to_add, to_update, to_remove, text_cache = diff_items(items, {})
    assert "A1" in text_cache
    assert "Paper One" in text_cache["A1"]
    assert "An abstract" in text_cache["A1"]


def test_diff_items_skips_empty_key():
    items = [{"key": "", "title": "No Key", "creators": [], "abstract": "", "tags": []}]
    to_add, to_update, to_remove, text_cache = diff_items(items, {})
    assert to_add == []
    assert text_cache == {}


def test_diff_items_mixed_operations():
    """Test a mix of add, update, remove, and unchanged."""
    item_a = _make_item("A1", "New Paper")
    item_b = _make_item("B1", "Updated Paper")
    item_c = _make_item("C1", "Unchanged")
    text_c = build_document_text(item_c)
    registry = {
        "B1": "old_hash",
        "C1": compute_hash(text_c),
        "D1": "removed_hash",
    }
    items = [item_a, item_b, item_c]
    to_add, to_update, to_remove, text_cache = diff_items(items, registry)
    assert len(to_add) == 1 and to_add[0]["key"] == "A1"
    assert len(to_update) == 1 and to_update[0]["key"] == "B1"
    assert to_remove == ["D1"]
    assert "A1" in text_cache
    assert "B1" in text_cache
    assert "C1" in text_cache
