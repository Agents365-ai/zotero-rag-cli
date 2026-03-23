from unittest.mock import patch, MagicMock

from rak.mcp_server import (
    mcp, search_papers, search_papers_bm25, similar_papers,
    ask_papers, export_papers, show_config, index_status,
)


def test_mcp_server_loads():
    assert mcp is not None
    assert mcp.name == "rak"


def test_index_status_no_index(tmp_path):
    from rak.config import RakConfig
    fake_config = RakConfig(data_dir=tmp_path)
    with patch("rak.mcp_server._get_config", return_value=fake_config):
        result = index_status()
    assert "No index found" in result


def test_index_status_with_index(tmp_path):
    import json
    from rak.config import RakConfig
    from rak.metadata import save_metadata
    fake_config = RakConfig(data_dir=tmp_path)
    save_metadata(tmp_path, "all-MiniLM-L6-v2", 42)
    with patch("rak.mcp_server._get_config", return_value=fake_config):
        result = index_status()
    data = json.loads(result)
    assert data["item_count"] == 42
    assert data["model_name"] == "all-MiniLM-L6-v2"


def test_search_papers_returns_json():
    import json
    from rak.searcher import SearchResult

    mock_searcher = MagicMock()
    mock_searcher.vector_search.return_value = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector"),
    ]
    mock_bm25 = MagicMock()

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, None, mock_bm25)):
        result = search_papers("test query", limit=5)

    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["key"] == "A1"
    assert data[0]["title"] == "Paper One"
    assert "text" not in data[0]


def test_search_papers_bm25_returns_json(tmp_path):
    import json
    from rak.config import RakConfig
    from rak.searcher import SearchResult

    fake_config = RakConfig(data_dir=tmp_path)

    mock_searcher = MagicMock()
    mock_searcher.bm25_search.return_value = [
        SearchResult(doc_id="B1", score=5.0, title="Keyword Paper", source="bm25", snippet="matched text"),
    ]

    with patch("rak.mcp_server._get_config", return_value=fake_config), \
         patch("rak.bm25.BM25Index.__init__", return_value=None), \
         patch("rak.bm25.BM25Index.close"), \
         patch("rak.searcher.Searcher.__init__", return_value=None), \
         patch("rak.searcher.Searcher.bm25_search", return_value=mock_searcher.bm25_search.return_value):
        result = search_papers_bm25("keyword query", limit=5)

    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["key"] == "B1"
    assert data[0]["source"] == "bm25"
    assert data[0]["snippet"] == "matched text"


def test_similar_papers_returns_json():
    import json
    from rak.searcher import SearchResult

    mock_searcher = MagicMock()
    mock_searcher.similar_search.return_value = [
        SearchResult(doc_id="C1", score=0.85, title="Similar Paper", source="similar"),
    ]
    mock_bm25 = MagicMock()

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, None, mock_bm25)):
        result = similar_papers("KEY1", limit=5)

    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["key"] == "C1"
    assert data[0]["source"] == "similar"
    mock_searcher.similar_search.assert_called_once_with("KEY1", limit=5, collection=None, tags=None)


def test_ask_papers_returns_answer():
    import json
    from rak.searcher import SearchResult

    mock_searcher = MagicMock()
    mock_searcher.vector_search.return_value = [
        SearchResult(doc_id="D1", score=0.9, title="Context Paper", source="vector", snippet="some context"),
    ]
    mock_bm25 = MagicMock()
    mock_llm = MagicMock()
    mock_llm.ask.return_value = "The answer is 42."

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, None, mock_bm25)), \
         patch("rak.llm.LLMClient", return_value=mock_llm):
        result = ask_papers("What is the answer?")

    data = json.loads(result)
    assert data["answer"] == "The answer is 42."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["key"] == "D1"


def test_ask_papers_no_results():
    import json

    mock_searcher = MagicMock()
    mock_searcher.vector_search.return_value = []
    mock_bm25 = MagicMock()

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, None, mock_bm25)):
        result = ask_papers("unanswerable question")

    data = json.loads(result)
    assert "No relevant papers" in data["answer"]
    assert data["sources"] == []


def test_export_papers_csv():
    from rak.searcher import SearchResult

    mock_searcher = MagicMock()
    mock_searcher.vector_search.return_value = [
        SearchResult(doc_id="E1", score=0.8, title="Export Paper", source="vector"),
    ]
    mock_vs = MagicMock()
    mock_vs.get.return_value = {"ids": ["E1"], "metadatas": [{"date": "2024", "item_type": "article"}]}
    mock_bm25 = MagicMock()

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, mock_vs, mock_bm25)):
        result = export_papers("test query")

    assert "key,title,score,source" in result
    assert "E1" in result


def test_export_papers_bibtex():
    from rak.searcher import SearchResult

    mock_searcher = MagicMock()
    mock_searcher.vector_search.return_value = [
        SearchResult(doc_id="F1", score=0.7, title="BibTeX Paper", source="vector"),
    ]
    mock_vs = MagicMock()
    mock_vs.get.return_value = {"ids": ["F1"], "metadatas": [{"date": "2024", "item_type": "conferencePaper"}]}
    mock_bm25 = MagicMock()

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, mock_vs, mock_bm25)):
        result = export_papers("test query", format="bibtex")

    assert "@inproceedings{F1," in result


def test_show_config(tmp_path):
    import json
    from rak.config import RakConfig

    fake_config = RakConfig(data_dir=tmp_path)
    with patch("rak.mcp_server._get_config", return_value=fake_config):
        result = show_config()

    data = json.loads(result)
    assert data["model_name"] == "all-MiniLM-L6-v2"
    assert data["pdf_provider"] == "pymupdf"
    assert data["chunk_size"] == 512
