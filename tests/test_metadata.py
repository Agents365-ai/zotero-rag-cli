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


def test_load_corrupt_json_returns_none(tmp_path: Path):
    (tmp_path / "meta.json").write_text("not valid json {{{")
    assert load_metadata(tmp_path) is None


def test_load_missing_keys_returns_none(tmp_path: Path):
    (tmp_path / "meta.json").write_text(json.dumps({"model_name": "test"}))
    assert load_metadata(tmp_path) is None


def test_save_metadata_atomic_write(tmp_path: Path):
    """save_metadata uses atomic write — no leftover tmp files on success."""
    save_metadata(tmp_path, model_name="test-model", item_count=10)
    meta_path = tmp_path / "meta.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert data["model_name"] == "test-model"
    assert data["item_count"] == 10
    assert "last_indexed" in data
    # No leftover .tmp files
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_save_metadata_overwrites_existing(tmp_path: Path):
    save_metadata(tmp_path, model_name="model-v1", item_count=100)
    save_metadata(tmp_path, model_name="model-v2", item_count=200)
    meta = load_metadata(tmp_path)
    assert meta.model_name == "model-v2"
    assert meta.item_count == 200
