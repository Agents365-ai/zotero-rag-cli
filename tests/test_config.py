from pathlib import Path
from unittest.mock import patch

from rak.config import RakConfig, detect_zotero_storage


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
