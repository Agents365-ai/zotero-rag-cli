from unittest.mock import patch

from click.testing import CliRunner

from rak.cli import main


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert "0.1.0" in result.output


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "Semantic search" in result.output
    assert result.exit_code == 0


def test_search_help():
    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert "--hybrid" in result.output
    assert "--limit" in result.output


def test_index_help():
    runner = CliRunner()
    result = runner.invoke(main, ["index", "--help"])
    assert result.exit_code == 0


def test_index_zot_not_found():
    runner = CliRunner()
    with patch("rak.indexer.shutil.which", return_value=None):
        result = runner.invoke(main, ["index"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_index_empty_library():
    runner = CliRunner()
    with patch("rak.indexer.shutil.which", return_value="/usr/bin/zot"), \
         patch("rak.indexer.subprocess.run") as mock_run, \
         patch("rak.embedder.SentenceTransformer"):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "[]"
        result = runner.invoke(main, ["index"])
    assert result.exit_code == 0
    assert "No items found" in result.output


def test_search_model_download_error():
    runner = CliRunner()
    with patch("rak.embedder.SentenceTransformer", side_effect=OSError("Network error")):
        result = runner.invoke(main, ["search", "test query"])
    assert result.exit_code == 1
    assert "Failed to load model" in result.output


import json as json_mod
from pathlib import Path


def test_index_writes_metadata(tmp_path: Path):
    from rak.config import RakConfig
    runner = CliRunner()
    zot_item = json_mod.dumps([{
        "key": "ABC123",
        "title": "Test Paper",
        "creators": [],
        "abstract": "An abstract.",
        "date": "2024",
        "tags": [],
        "item_type": "journalArticle",
    }])
    fake_config = RakConfig(data_dir=tmp_path)
    with patch("rak.indexer.shutil.which", return_value="/usr/bin/zot"), \
         patch("rak.indexer.subprocess.run") as mock_run, \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.cli.RakConfig", return_value=fake_config):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = zot_item
        mock_model = mock_st.return_value
        mock_model.encode.return_value.__iter__ = lambda self: iter([0.1] * 384)
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384
        mock_model.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["index"])

    meta_path = tmp_path / "meta.json"
    assert meta_path.exists(), f"meta.json not found; output was: {result.output}"
    meta = json_mod.loads(meta_path.read_text())
    assert meta["item_count"] >= 1
    assert meta["model_name"] == "all-MiniLM-L6-v2"
