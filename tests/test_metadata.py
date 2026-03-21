import json
from pathlib import Path

from rak.metadata import IndexMetadata, save_metadata, load_metadata


def test_save_and_load_roundtrip(tmp_path: Path):
    save_metadata(tmp_path, model_name="all-MiniLM-L6-v2", item_count=342)
    meta = load_metadata(tmp_path)
    assert meta is not None
    assert meta.model_name == "all-MiniLM-L6-v2"
    assert meta.item_count == 342
    assert meta.last_indexed  # non-empty ISO timestamp


def test_load_missing_returns_none(tmp_path: Path):
    assert load_metadata(tmp_path) is None
