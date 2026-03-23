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


def test_load_corrupt_json_returns_empty(tmp_path: Path):
    (tmp_path / "registry.json").write_text("{corrupt json!!!")
    assert load_registry(tmp_path) == {}


def test_load_non_dict_returns_empty(tmp_path: Path):
    """If registry.json contains a list instead of dict, return empty."""
    import json
    (tmp_path / "registry.json").write_text(json.dumps(["a", "b"]))
    assert load_registry(tmp_path) == {}


def test_load_coerces_keys_and_values_to_strings(tmp_path: Path):
    import json
    (tmp_path / "registry.json").write_text(json.dumps({123: 456}))
    result = load_registry(tmp_path)
    assert result == {"123": "456"}


def test_save_registry_atomic_write(tmp_path: Path):
    """save_registry uses atomic write — no leftover tmp files on success."""
    registry = {"K1": "hash1", "K2": "hash2"}
    save_registry(tmp_path, registry)
    target = tmp_path / "registry.json"
    assert target.exists()
    import json
    assert json.loads(target.read_text()) == registry
    # No leftover .tmp files
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_save_registry_overwrites_existing(tmp_path: Path):
    import json
    save_registry(tmp_path, {"A": "1"})
    save_registry(tmp_path, {"B": "2"})
    loaded = load_registry(tmp_path)
    assert loaded == {"B": "2"}
