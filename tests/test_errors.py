from rak.errors import RakError, ZotNotFoundError, EmptyLibraryError, ModelDownloadError


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
