from pathlib import Path
from unittest.mock import patch

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
