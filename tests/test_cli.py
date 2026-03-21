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
