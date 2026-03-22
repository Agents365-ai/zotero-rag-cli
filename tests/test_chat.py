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
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector",
                     snippet="Full text of paper one."),
    ]
    searcher = _mock_searcher(results)
    llm = _mock_llm([])

    session = ChatSession(searcher=searcher, llm=llm)
    session.search("test query")

    assert len(session.context) == 1
    assert session.context[0]["key"] == "A1"
    assert session.context[0]["text"] == "Full text of paper one."
    assert len(session.messages) == 1  # system message only
    assert session.messages[0]["role"] == "system"


def test_chat_session_ask_appends_history():
    results = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector",
                     snippet="Text."),
    ]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["The ", "answer."])

    session = ChatSession(searcher=searcher, llm=llm)
    session.search("query")

    tokens = list(session.ask("What is this about?"))
    assert tokens == ["The ", "answer."]
    assert len(session.messages) == 3  # system + user + assistant
    assert session.messages[1]["role"] == "user"
    assert session.messages[1]["content"] == "What is this about?"
    assert session.messages[2]["role"] == "assistant"
    assert session.messages[2]["content"] == "The answer."


def test_chat_session_search_resets_history():
    results = [
        SearchResult(doc_id="A1", score=0.9, title="Paper One", source="vector",
                     snippet="Text."),
    ]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["Response."])

    session = ChatSession(searcher=searcher, llm=llm)

    session.search("first query")
    list(session.ask("question"))  # adds user + assistant messages

    assert len(session.messages) == 3

    # Re-search should reset
    llm.stream_messages.return_value = iter(["New."])
    session.search("second query")
    assert len(session.messages) == 1  # only system message
    assert session.messages[0]["role"] == "system"


def test_estimate_tokens():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0
    # ~4 chars per token
    assert estimate_tokens("a" * 400) == 100


def test_token_count_property():
    results = [SearchResult(doc_id="A1", score=0.9, title="P", source="vector", snippet="Text.")]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["Reply."])
    session = ChatSession(searcher=searcher, llm=llm)
    session.search("query")
    initial_tokens = session.token_count
    assert initial_tokens > 0
    list(session.ask("question"))
    assert session.token_count > initial_tokens


def test_turn_count_property():
    results = [SearchResult(doc_id="A1", score=0.9, title="P", source="vector", snippet="T.")]
    searcher = _mock_searcher(results)
    llm = _mock_llm(["R."])
    session = ChatSession(searcher=searcher, llm=llm)
    session.search("q")
    assert session.turn_count == 0
    list(session.ask("q1"))
    assert session.turn_count == 1


def test_help_text_contains_commands():
    assert "/search" in HELP_TEXT
    assert "/context" in HELP_TEXT
    assert "/tokens" in HELP_TEXT
    assert "/help" in HELP_TEXT
    assert "/quit" in HELP_TEXT


def test_chat_session_bm25_only_mode():
    results = [
        SearchResult(doc_id="K1", score=5.0, title="", source="bm25", snippet="keyword match"),
    ]
    searcher = MagicMock()
    searcher.bm25_search.return_value = results
    llm = _mock_llm(["Answer."])

    session = ChatSession(searcher=searcher, llm=llm, bm25_only=True)
    session.search("keyword query")

    assert len(session.context) == 1
    assert session.context[0]["key"] == "K1"
    searcher.bm25_search.assert_called_once()
    searcher.vector_search.assert_not_called()
    searcher.hybrid_search.assert_not_called()
