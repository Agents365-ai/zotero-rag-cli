from pathlib import Path
from unittest.mock import patch, MagicMock

from rak.indexer import diff_items


def test_diff_items_passes_pdf_provider(tmp_path: Path):
    """diff_items passes pdf_provider to extract_file_text."""
    items = [{"key": "ABC", "title": "Test Paper"}]
    registry = {}
    storage_dir = tmp_path
    key_dir = storage_dir / "ABC"
    key_dir.mkdir()
    (key_dir / "paper.pdf").write_bytes(b"%PDF")

    with patch("rak.indexer.extract_file_text", return_value="extracted") as mock_extract:
        diff_items(items, registry, storage_dir=storage_dir, pdf_provider="mineru")

    mock_extract.assert_called()
    call_kwargs = mock_extract.call_args
    assert call_kwargs[1].get("provider") == "mineru" or \
           (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "mineru")
