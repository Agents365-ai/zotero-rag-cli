from __future__ import annotations

from openai import OpenAI, APIConnectionError

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


class LLMClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model
        self._base_url = base_url

    def ask(self, question: str, context: list[dict]) -> str:
        context_parts = []
        for i, doc in enumerate(context, 1):
            context_parts.append(
                f"[{i}] {doc['key']} - {doc['title']}\n{doc['text']}"
            )
        context_text = "\n\n".join(context_parts) if context_parts else "No papers found."

        user_message = f"Papers:\n\n{context_text}\n\nQuestion: {question}"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except APIConnectionError:
            raise LLMConnectionError(self._base_url)
