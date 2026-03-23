from unittest.mock import MagicMock, patch

import pytest

from rak.llm import LLMClient, LLMConnectionError


def test_ask_builds_prompt_and_returns_response():
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        context = [
            {"key": "A1", "title": "Paper One", "text": "Content of paper one."},
            {"key": "A2", "title": "Paper Two", "text": "Content of paper two."},
        ]
        answer = client.ask("What is the meaning?", context)

        assert answer == "The answer is 42."
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "research" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "What is the meaning?" in messages[1]["content"]
        assert "Paper One" in messages[1]["content"]
        assert "Paper Two" in messages[1]["content"]


def test_ask_connection_error():
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        from openai import APIConnectionError
        mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        with pytest.raises(LLMConnectionError, match="not reachable"):
            client.ask("test", [])


def test_ask_stream_yields_tokens():
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        # Simulate streaming chunks
        chunks = []
        for text in ["The ", "answer ", "is ", "42."]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        mock_client.chat.completions.create.return_value = iter(chunks)

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        context = [{"key": "A1", "title": "Paper", "text": "Content."}]
        tokens = list(client.ask_stream("question", context))

        assert tokens == ["The ", "answer ", "is ", "42."]
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["stream"] is True


def test_ask_stream_connection_error():
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        from openai import APIConnectionError
        mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        with pytest.raises(LLMConnectionError, match="not reachable"):
            list(client.ask_stream("test", []))


def test_ask_server_error():
    from rak.llm import LLMServerError
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        from openai import APIStatusError
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "model not found"}
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="model not found", response=mock_response, body=None,
        )

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        with pytest.raises(LLMServerError, match="LLM server error"):
            client.ask("test", [])


def test_ask_stream_server_error():
    from rak.llm import LLMServerError
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        from openai import APIStatusError
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {}
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="not found", response=mock_response, body=None,
        )

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        with pytest.raises(LLMServerError):
            list(client.ask_stream("test", []))


def test_ask_stream_empty_delta():
    """Chunks with no content should be skipped."""
    with patch("rak.llm.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None  # no content
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "Hello"
        chunk3 = MagicMock()
        chunk3.choices = []  # no choices
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
        tokens = list(client.ask_stream("q", [{"key": "A", "title": "T", "text": "C"}]))
        assert tokens == ["Hello"]


def test_build_messages_format():
    """Verify message structure."""
    with patch("rak.llm.OpenAI"):
        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
    context = [{"key": "A1", "title": "Paper", "text": "Content"}]
    messages = client._build_messages("What is X?", context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "[1] A1 - Paper" in messages[1]["content"]
    assert "What is X?" in messages[1]["content"]


def test_build_messages_empty_context():
    with patch("rak.llm.OpenAI"):
        client = LLMClient(base_url="http://localhost:11434/v1", model="llama3")
    messages = client._build_messages("question", [])
    assert "No papers found" in messages[1]["content"]
