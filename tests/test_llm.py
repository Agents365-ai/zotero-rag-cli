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
