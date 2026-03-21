from unittest.mock import patch
from rak.errors import RakError, ZotNotFoundError, EmptyLibraryError, ModelDownloadError
from rak.indexer import fetch_zot_items
from rak.embedder import Embedder
import pytest


def test_zot_not_found_is_rak_error():
    err = ZotNotFoundError("zot")
    assert isinstance(err, RakError)
    assert "zot" in str(err)


def test_empty_library_is_rak_error():
    err = EmptyLibraryError()
    assert isinstance(err, RakError)


def test_model_download_error_is_rak_error():
    err = ModelDownloadError("all-MiniLM-L6-v2", "Connection refused")
    assert isinstance(err, RakError)
    assert "all-MiniLM-L6-v2" in str(err)
    assert "Connection refused" in str(err)


def test_fetch_zot_items_raises_zot_not_found():
    with patch("shutil.which", return_value=None):
        with pytest.raises(ZotNotFoundError):
            fetch_zot_items("zot")


def test_fetch_zot_items_raises_empty_library():
    with patch("shutil.which", return_value="/usr/bin/zot"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "[]"
        with pytest.raises(EmptyLibraryError):
            fetch_zot_items("zot")


def test_embedder_raises_model_download_error():
    with patch("rak.embedder.SentenceTransformer", side_effect=OSError("Connection refused")):
        with pytest.raises(ModelDownloadError, match="Connection refused"):
            Embedder("bad-model-name")
