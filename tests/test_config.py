from pathlib import Path
from unittest.mock import patch

import pytest

from rak.config import RakConfig, detect_zotero_storage, save_config, load_config


def test_detect_zotero_storage_found(tmp_path: Path):
    storage_dir = tmp_path / "Zotero" / "storage"
    storage_dir.mkdir(parents=True)
    with patch("rak.config.Path.home", return_value=tmp_path):
        result = detect_zotero_storage()
    assert result == storage_dir


def test_detect_zotero_storage_missing(tmp_path: Path):
    with patch("rak.config.Path.home", return_value=tmp_path):
        result = detect_zotero_storage()
    assert result is None


def test_rak_config_has_zotero_storage():
    config = RakConfig()
    assert config.zotero_storage_dir is None or isinstance(config.zotero_storage_dir, Path)


def test_save_and_load_config(tmp_path: Path):
    save_config(tmp_path, "llm_model", "mistral")
    cfg = load_config(tmp_path)
    assert cfg["llm_model"] == "mistral"


def test_save_config_merges(tmp_path: Path):
    save_config(tmp_path, "llm_model", "mistral")
    save_config(tmp_path, "llm_base_url", "http://localhost:1234/v1")
    cfg = load_config(tmp_path)
    assert cfg["llm_model"] == "mistral"
    assert cfg["llm_base_url"] == "http://localhost:1234/v1"


def test_load_config_missing(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert cfg == {}


def test_save_config_coerces_chunk_size_to_int(tmp_path: Path):
    save_config(tmp_path, "chunk_size", "256")
    cfg = load_config(tmp_path)
    assert cfg["chunk_size"] == 256
    assert isinstance(cfg["chunk_size"], int)


def test_save_config_rejects_invalid_chunk_size(tmp_path: Path):
    with pytest.raises(ValueError, match="expected int"):
        save_config(tmp_path, "chunk_size", "not_a_number")


def test_save_config_rejects_malicious_zot_command(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid zot_command"):
        save_config(tmp_path, "zot_command", "/usr/bin/evil")


def test_save_config_accepts_simple_zot_command(tmp_path: Path):
    save_config(tmp_path, "zot_command", "zot")
    cfg = load_config(tmp_path)
    assert cfg["zot_command"] == "zot"


def test_config_loads_chunk_size_as_int(tmp_path: Path):
    import json
    # Simulate legacy config with string chunk_size
    (tmp_path / "config.json").write_text(json.dumps({"chunk_size": "128"}))
    config = RakConfig(data_dir=tmp_path)
    assert config.chunk_size == 128
    assert isinstance(config.chunk_size, int)
