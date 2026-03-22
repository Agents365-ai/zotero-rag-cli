from unittest.mock import patch

from click.testing import CliRunner

from rak.cli import main


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    from rak import __version__
    assert __version__ in result.output


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
from rak.metadata import save_metadata


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


def test_status_no_index(tmp_path: Path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "No index found" in result.output


def test_status_with_index(tmp_path: Path):
    from rak.config import RakConfig
    save_metadata(tmp_path, model_name="all-MiniLM-L6-v2", item_count=342)
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "342" in result.output
    assert "all-MiniLM-L6-v2" in result.output


def test_status_json_output(tmp_path: Path):
    from rak.config import RakConfig
    save_metadata(tmp_path, model_name="all-MiniLM-L6-v2", item_count=100)
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["--json", "status"])
    data = json_mod.loads(result.output)
    assert data["item_count"] == 100
    assert data["model_name"] == "all-MiniLM-L6-v2"


def test_clear_with_yes(tmp_path: Path):
    from rak.config import RakConfig
    # Create fake index files
    (tmp_path / "chroma").mkdir()
    (tmp_path / "fts.sqlite").touch()
    save_metadata(tmp_path, model_name="test", item_count=10)

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["clear", "--yes"])
    assert result.exit_code == 0
    assert "Cleared" in result.output
    assert not (tmp_path / "chroma").exists()
    assert not (tmp_path / "fts.sqlite").exists()
    assert not (tmp_path / "meta.json").exists()


def test_clear_nothing_to_clear(tmp_path: Path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["clear", "--yes"])
    assert result.exit_code == 0
    assert "Nothing to clear" in result.output


def test_clear_prompts_without_yes(tmp_path: Path):
    from rak.config import RakConfig
    (tmp_path / "chroma").mkdir()
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["clear"], input="n\n")
    assert (tmp_path / "chroma").exists()  # not deleted because user said no


def test_index_help_shows_full_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["index", "--help"])
    assert "--full" in result.output


def test_search_help_shows_filter_flags():
    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert "--collection" in result.output
    assert "--tag" in result.output


def test_ask_help():
    runner = CliRunner()
    result = runner.invoke(main, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--context" in result.output
    assert "--hybrid" in result.output
    assert "--llm-model" in result.output
    assert "--llm-url" in result.output
    assert "--collection" in result.output
    assert "--tag" in result.output


def test_config_help():
    runner = CliRunner()
    result = runner.invoke(main, ["config", "--help"])
    assert result.exit_code == 0
    assert "Show or set" in result.output


def test_export_help():
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    assert "--output" in result.output
    assert "--hybrid" in result.output
    assert "--collection" in result.output
    assert "--tag" in result.output


def test_chat_help():
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "Interactive multi-turn" in result.output
    assert "--context" in result.output
    assert "--hybrid" in result.output
    assert "--llm-model" in result.output


def test_completion_help():
    runner = CliRunner()
    result = runner.invoke(main, ["completion", "--help"])
    assert result.exit_code == 0
    assert "shell completion" in result.output.lower()
