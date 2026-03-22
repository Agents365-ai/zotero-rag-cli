from pathlib import Path

from rak.registry import load_registry, save_registry, compute_hash


def test_save_and_load_roundtrip(tmp_path: Path):
    registry = {"ABC123": "a1b2c3", "DEF456": "d4e5f6"}
    save_registry(tmp_path, registry)
    loaded = load_registry(tmp_path)
    assert loaded == registry


def test_load_missing_returns_empty_dict(tmp_path: Path):
    assert load_registry(tmp_path) == {}


def test_compute_hash_deterministic():
    text = "Single Cell Analysis\nAuthors: John Doe"
    assert compute_hash(text) == compute_hash(text)


def test_compute_hash_different_texts():
    assert compute_hash("hello") != compute_hash("world")
