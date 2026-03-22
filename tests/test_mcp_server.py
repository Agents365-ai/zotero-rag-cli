from unittest.mock import patch, MagicMock

from rak.mcp_server import mcp, search_papers, index_status


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


def test_search_papers_returns_json_with_text():
    import json
    from rak.searcher import SearchResult

    mock_searcher = MagicMock()
    mock_searcher.vector_search.return_value = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector"),
    ]
    mock_store = MagicMock()
    mock_store._collection.get.return_value = {"documents": ["Full text of paper."]}
    mock_bm25 = MagicMock()

    with patch("rak.mcp_server._get_config"), \
         patch("rak.mcp_server._init_searcher", return_value=(mock_searcher, mock_store, mock_bm25)):
        result = search_papers("test query", limit=5)

    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["key"] == "A1"
    assert data[0]["title"] == "Paper One"
    assert data[0]["text"] == "Full text of paper."
