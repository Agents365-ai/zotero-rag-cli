from unittest.mock import MagicMock, patch

from rak.chat import ChatSession, estimate_tokens, HELP_TEXT
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
    mock_store.get.return_value = {"ids": ["A1"], "documents": ["Full text of paper one."]}
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
    mock_store.get.return_value = {"ids": ["A1"], "documents": ["Text."]}
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
    mock_store.get.return_value = {"ids": ["A1"], "documents": ["Text."]}

    session.search("first query", vector_store=mock_store)
    list(session.ask("question"))  # adds user + assistant messages

    assert len(session.messages) == 3

    # Re-search should reset
    llm.stream_messages.return_value = iter(["New."])
    session.search("second query", vector_store=mock_store)
    assert len(session.messages) == 1  # only system message
    assert session.messages[0]["role"] == "system"


def test_estimate_tokens():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0
    # ~4 chars per token
    assert estimate_tokens("a" * 400) == 100


def test_token_count_property():
    results = [SearchResult(doc_id="A1", score=0.9, title="P", source="vector")]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["Reply."])
    session = ChatSession(searcher=searcher, llm=llm)
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["A1"], "documents": ["Text."]}
    session.search("query", vector_store=mock_store)
    initial_tokens = session.token_count
    assert initial_tokens > 0
    list(session.ask("question"))
    assert session.token_count > initial_tokens


def test_turn_count_property():
    results = [SearchResult(doc_id="A1", score=0.9, title="P", source="vector")]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["R."])
    session = ChatSession(searcher=searcher, llm=llm)
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["A1"], "documents": ["T."]}
    session.search("q", vector_store=mock_store)
    assert session.turn_count == 0
    list(session.ask("q1"))
    assert session.turn_count == 1


def test_help_text_contains_commands():
    assert "/search" in HELP_TEXT
    assert "/context" in HELP_TEXT
    assert "/tokens" in HELP_TEXT
    assert "/help" in HELP_TEXT
    assert "/quit" in HELP_TEXT
