from unittest.mock import patch, MagicMock

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
         patch("sentence_transformers.SentenceTransformer"):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "[]"
        result = runner.invoke(main, ["index"])
    assert result.exit_code == 0
    assert "No items found" in result.output


def test_search_model_download_error():
    runner = CliRunner()
    with patch("sentence_transformers.SentenceTransformer", side_effect=OSError("Network error")):
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
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
    assert "--bm25" in result.output
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
    assert "--bm25" in result.output
    assert "--collection" in result.output
    assert "--tag" in result.output


def test_chat_help():
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "Interactive multi-turn" in result.output
    assert "--context" in result.output
    assert "--hybrid" in result.output
    assert "--bm25" in result.output
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.get", return_value={"ids": ["A1", "A2"], "metadatas": [{"date": "2024", "authors": "Doe"}, {"date": "2024", "authors": "Doe"}]}), \
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.get", return_value={"ids": ["B1"], "metadatas": [{"date": "2024", "authors": "Smith"}]}), \
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.get", return_value={"ids": ["C1"], "metadatas": [{}]}), \
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
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
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=[]), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["--json", "search", "nothing"])

    assert result.exit_code == 0
    data = json_mod.loads(result.output)
    assert data == []


def test_search_bm25_only(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="K1", score=5.0, title="", source="bm25", snippet="keyword match"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.searcher.Searcher.bm25_search", return_value=mock_results), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        result = runner.invoke(main, ["search", "test", "--bm25"])

    assert result.exit_code == 0
    assert "K1" in result.output


def test_search_bm25_help_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert "--bm25" in result.output


def test_search_hybrid_mode(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="H1", score=0.03, title="Hybrid Paper", source="fused"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.hybrid_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["search", "test", "--hybrid"])

    assert result.exit_code == 0
    assert "Hybrid Paper" in result.output


# --- similar command tests ---


def test_similar_help():
    runner = CliRunner()
    result = runner.invoke(main, ["similar", "--help"])
    assert result.exit_code == 0
    assert "KEY_OR_TITLE" in result.output
    assert "--limit" in result.output


def test_similar_by_key(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="B1", score=0.85, title="Similar Paper", source="similar"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.has", return_value=True), \
         patch("rak.searcher.Searcher.similar_search", return_value=mock_results), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["similar", "ABC12345"])

    assert result.exit_code == 0
    assert "Similar Paper" in result.output


def test_similar_no_results(tmp_path: Path):
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.store.VectorStore.has", return_value=True), \
         patch("rak.searcher.Searcher.similar_search", return_value=[]), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["similar", "ABC12345"])

    assert result.exit_code == 0
    assert "No similar papers" in result.output


# --- reindex command tests ---


def test_reindex_help():
    runner = CliRunner()
    result = runner.invoke(main, ["reindex", "--help"])
    assert result.exit_code == 0
    assert "Clear indexes and rebuild" in result.output


def test_reindex_zot_not_found(tmp_path: Path):
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.indexer.shutil.which", return_value=None):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["reindex"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_reindex_empty_library(tmp_path: Path):
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.indexer.shutil.which", return_value="/usr/bin/zot"), \
         patch("rak.indexer.subprocess.run") as mock_run:
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "[]"
        result = runner.invoke(main, ["reindex"])

    assert result.exit_code == 0
    assert "No items found" in result.output


# --- ask command tests ---


def test_ask_json_output(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="D1", score=0.9, title="Context Paper", source="vector", snippet="some context"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"), \
         patch("rak.llm.OpenAI") as mock_openai:
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        mock_client = mock_openai.return_value
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "The answer."
        mock_client.chat.completions.create.return_value = mock_resp
        result = runner.invoke(main, ["--json", "ask", "What is it?"])

    assert result.exit_code == 0
    data = json_mod.loads(result.output)
    assert data["answer"] == "The answer."
    assert len(data["sources"]) == 1


def test_ask_streaming_output(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="D1", score=0.9, title="Context Paper", source="vector", snippet="ctx"),
    ]

    # Build streaming chunks
    chunks = []
    for text in ["Hello", " world"]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"), \
         patch("rak.llm.OpenAI") as mock_openai:
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = iter(chunks)
        result = runner.invoke(main, ["ask", "What is it?"])

    assert result.exit_code == 0
    assert "Sources:" in result.output
    assert "D1" in result.output


def test_ask_no_results(tmp_path: Path):
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=[]), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        result = runner.invoke(main, ["ask", "anything"])

    assert result.exit_code == 0
    assert "No relevant papers" in result.output


def test_ask_llm_connection_error(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="D1", score=0.9, title="Paper", source="vector", snippet="ctx"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("rak.searcher.Searcher.vector_search", return_value=mock_results), \
         patch("rak.store.VectorStore.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"), \
         patch("rak.llm.OpenAI") as mock_openai:
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        from openai import APIConnectionError
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        result = runner.invoke(main, ["--json", "ask", "test"])

    assert result.exit_code == 1
    assert "not reachable" in result.output


# --- _resolve_key tests ---

from rak.cli import _resolve_key


def test_resolve_key_direct_match():
    mock_vs = MagicMock()
    mock_vs.has.return_value = True
    mock_bm25 = MagicMock()

    result = _resolve_key("ABC12345", mock_vs, mock_bm25)
    assert result == "ABC12345"
    mock_bm25.search.assert_not_called()


def test_resolve_key_not_found_in_store():
    """Key pattern but not in store -> falls back to title search."""
    mock_vs = MagicMock()
    mock_vs.has.return_value = False  # not found for either key or chunk_0
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = []

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = _resolve_key("ZZZZZZZZ", mock_vs, mock_bm25)
    assert result is None


def test_resolve_key_title_single_match():
    """Title search returning one result -> auto-select."""
    mock_vs = MagicMock()
    mock_vs.has.return_value = False
    mock_vs.get.return_value = {
        "ids": ["KEY1"],
        "metadatas": [{"title": "Attention Is All You Need"}],
    }
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [{"id": "KEY1", "score": 5.0}]

    result = _resolve_key("attention is all you need", mock_vs, mock_bm25)
    assert result == "KEY1"


# --- completion command test ---


def test_completion_runs():
    """completion command should invoke subprocess."""
    runner = CliRunner()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "# completion script"
        mock_run.return_value.returncode = 0
        result = runner.invoke(main, ["completion", "bash"])
    assert result.exit_code == 0


# --- export --bm25 tests ---


def test_export_bm25_csv(tmp_path: Path):
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()

    mock_results = [
        SearchResult(doc_id="BM1", score=5.0, title="BM25 Paper", source="bm25"),
    ]

    with patch("rak.cli.RakConfig", return_value=fake_config), \
         patch("rak.searcher.Searcher.bm25_search", return_value=mock_results), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"):
        result = runner.invoke(main, ["export", "test", "--bm25"])

    assert result.exit_code == 0
    assert "BM1" in result.output
    assert "BM25 Paper" in result.output


# --- config embedding_provider=api display ---


def test_config_show_api_embedding(tmp_path: Path):
    from rak.config import RakConfig, save_config

    save_config(tmp_path, "embedding_provider", "api")
    save_config(tmp_path, "embedding_base_url", "http://example.com/v1")
    save_config(tmp_path, "embedding_api_key", "sk-test12345678")
    fake_config = RakConfig(data_dir=tmp_path)
    runner = CliRunner()
    with patch("rak.cli.RakConfig", return_value=fake_config):
        result = runner.invoke(main, ["config"])

    assert result.exit_code == 0
    assert "embedding_provider = api" in result.output
    assert "embedding_base_url = " in result.output
    assert "embedding_api_key = " in result.output
    # Key should be masked
    assert "sk-test12345678" not in result.output
