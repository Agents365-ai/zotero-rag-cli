from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI, APIConnectionError, APIStatusError

from rak.errors import RakError

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the following question based on "
    "the provided research papers. Cite papers by their key (e.g., [ABC123]) "
    "when referencing them. If the papers don't contain enough information "
    "to answer, say so."
)


class LLMConnectionError(RakError):
    def __init__(self, base_url: str) -> None:
        super().__init__(
            f"LLM server not reachable at {base_url}. "
            "Start Ollama or LMStudio first."
        )


class LLMServerError(RakError):
    def __init__(self, base_url: str, model: str, detail: str = "") -> None:
        msg = f"LLM server error at {base_url} (model: {model})."
        if detail:
            msg += f" {detail}"
        msg += " Check that the model is installed: ollama pull " + model
        super().__init__(msg)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed") -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._base_url = base_url

    def _build_messages(self, question: str, context: list[dict]) -> list[dict]:
        context_parts = []
        for i, doc in enumerate(context, 1):
            context_parts.append(
                f"[{i}] {doc['key']} - {doc['title']}\n{doc['text']}"
            )
        context_text = "\n\n".join(context_parts) if context_parts else "No papers found."
        user_message = f"Papers:\n\n{context_text}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def ask(self, question: str, context: list[dict]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(question, context),
            )
            return response.choices[0].message.content
        except APIConnectionError:
            raise LLMConnectionError(self._base_url)
        except APIStatusError as exc:
            raise LLMServerError(self._base_url, self._model, str(exc))

    def ask_stream(self, question: str, context: list[dict]) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(question, context),
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIConnectionError:
            raise LLMConnectionError(self._base_url)
        except APIStatusError as exc:
            raise LLMServerError(self._base_url, self._model, str(exc))

    def stream_messages(self, messages: list[dict]) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIConnectionError:
            raise LLMConnectionError(self._base_url)
        except APIStatusError as exc:
            raise LLMServerError(self._base_url, self._model, str(exc))
