from unittest.mock import MagicMock, patch

from rak.chat import ChatSession
from rak.searcher import SearchResult


def _mock_searcher(results):
    searcher = MagicMock()
    searcher.vector_search.return_value = results
    searcher.hybrid_search.return_value = results
    return searcher


def _mock_llm(tokens):
    llm = MagicMock()
    llm.stream_messages.return_value = iter(tokens)
    return llm


def test_chat_session_search_populates_context():
    results = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector"),
    ]
    searcher = _mock_searcher(results)
    llm = _mock_llm([])

    session = ChatSession(searcher=searcher, llm=llm)

    mock_store = MagicMock()
    mock_store._collection.get.return_value = {"documents": ["Full text of paper one."]}
    session.search("test query", vector_store=mock_store)

    assert len(session.context) == 1
    assert session.context[0]["key"] == "A1"
    assert session.context[0]["text"] == "Full text of paper one."
    assert len(session.messages) == 1  # system message only
    assert session.messages[0]["role"] == "system"


def test_chat_session_ask_appends_history():
    results = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector"),
    ]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["The ", "answer."])

    session = ChatSession(searcher=searcher, llm=llm)

    mock_store = MagicMock()
    mock_store._collection.get.return_value = {"documents": ["Text."]}
    session.search("query", vector_store=mock_store)

    tokens = list(session.ask("What is this about?"))
    assert tokens == ["The ", "answer."]
    assert len(session.messages) == 3  # system + user + assistant
    assert session.messages[1]["role"] == "user"
    assert session.messages[1]["content"] == "What is this about?"
    assert session.messages[2]["role"] == "assistant"
    assert session.messages[2]["content"] == "The answer."


def test_chat_session_search_resets_history():
    results = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector"),
    ]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["Response."])

    session = ChatSession(searcher=searcher, llm=llm)

    mock_store = MagicMock()
    mock_store._collection.get.return_value = {"documents": ["Text."]}

    session.search("first query", vector_store=mock_store)
    list(session.ask("question"))  # adds user + assistant messages

    assert len(session.messages) == 3

    # Re-search should reset
    llm.stream_messages.return_value = iter(["New."])
    session.search("second query", vector_store=mock_store)
    assert len(session.messages) == 1  # only system message
    assert session.messages[0]["role"] == "system"
