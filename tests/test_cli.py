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
        import numpy as np
        def _fake_encode(text_or_texts, **kwargs):
            if isinstance(text_or_texts, list):
                return np.array([[0.1] * 384 for _ in text_or_texts])
            return np.array([0.1] * 384)
        mock_model.encode.side_effect = _fake_encode
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


# --- Config integration tests ---


def test_config_show_all(tmp_path: Path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert "model_name = " in result.output
    assert "zot_command = " in result.output
    assert "llm_base_url = " in result.output
    assert "llm_model = " in result.output
    assert "llm_api_key = " in result.output


def test_config_get_single_key(tmp_path: Path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config", "llm_model"])
    assert result.exit_code == 0
    assert "llm_model = llama3" in result.output


def test_config_get_unknown_key(tmp_path: Path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config", "nonexistent_key"])
    assert result.exit_code == 1


def test_config_set_value(tmp_path: Path):
    from rak.config import RakConfig, load_config
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config", "llm_model", "mistral"])
    assert result.exit_code == 0
    assert "llm_model = mistral" in result.output
    cfg = load_config(tmp_path)
    assert cfg["llm_model"] == "mistral"


def test_config_set_unknown_key(tmp_path: Path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config", "bad_key", "value"])
    assert result.exit_code == 1
    assert "Unknown config key" in result.output


def test_config_masks_api_key(tmp_path: Path):
    from rak.config import RakConfig, save_config
    save_config(tmp_path, "llm_api_key", "sk-abcdefghijklmnop")
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config"])
    assert "..." in result.output
    assert "sk-abcdefghijklmnop" not in result.output


# --- Export integration tests ---


def test_export_csv_output(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector"),
        SearchResult(doc_id="A2", score=0.7, title="Paper Two", source="vector"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.get", return_value={"metadatas": [{"date": "2024", "authors": "Doe"}]}), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["export", "test query"])

    assert result.exit_code == 0
    assert "key,title,score,source" in result.output
    assert "A1" in result.output
    assert "Paper One" in result.output


def test_export_bibtex_output(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="B1", score=0.8, title="Deep Learning", source="vector"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.get", return_value={"metadatas": [{"date": "2024", "authors": "Smith"}]}), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["export", "test query", "--format", "bibtex"])

    assert result.exit_code == 0
    assert "@article{B1," in result.output
    assert "Deep Learning" in result.output


def test_export_to_file(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    output_file = str(tmp_path / "results.csv")

    mock_results = [
        SearchResult(doc_id="C1", score=0.5, title="Paper", source="vector"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.get", return_value={"metadatas": [{}]}), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["export", "test", "--output", output_file])

    assert result.exit_code == 0
    assert "Exported" in result.output
    content = Path(output_file).read_text()
    assert "C1" in content


def test_export_no_results(tmp_path: Path):
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=[]), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["export", "nonexistent"])

    assert result.exit_code == 0
    assert "No results found" in result.output


# --- Search integration tests ---


def test_search_vector_results(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="X1", score=0.95, title="RNA Sequencing", source="vector", snippet="chunk text"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["search", "RNA"])

    assert result.exit_code == 0
    assert "RNA Sequencing" in result.output


def test_search_json_output(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="X1", score=0.95, title="Paper", source="vector", snippet="text"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["--json", "search", "test"])

    assert result.exit_code == 0
    data = json_mod.loads(result.output)
    assert len(data) == 1
    assert data[0]["key"] == "X1"
    assert data[0]["snippet"] == "text"


def test_search_no_results_json(tmp_path: Path):
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=[]), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["--json", "search", "nothing"])

    assert result.exit_code == 0
    data = json_mod.loads(result.output)
    assert data == []


def test_search_hybrid_mode(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="H1", score=0.03, title="Hybrid Paper", source="fused"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.embedder.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.hybrid_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["search", "test", "--hybrid"])

    assert result.exit_code == 0
    assert "Hybrid Paper" in result.output
