from __future__ import annotations

from collections.abc import Iterator

from rak.llm import LLMClient, SYSTEM_PROMPT
from rak.searcher import Searcher


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


HELP_TEXT = """\
Commands:
  /search <query>  — Retrieve new papers and reset conversation
  /context         — Show current paper list
  /tokens          — Show estimated token usage
  /help            — Show this help message
  /quit            — Exit chat session"""


class ChatSession:
    def __init__(self, searcher: Searcher, llm: LLMClient, limit: int = 5,
                 collection: str | None = None, tags: list[str] | None = None,
                 hybrid: bool = False, bm25_only: bool = False) -> None:
        self._searcher = searcher
        self._llm = llm
        self._limit = limit
        self._collection = collection
        self._tags = tags
        self._hybrid = hybrid
        self._bm25_only = bm25_only
        self.context: list[dict] = []
        self.messages: list[dict] = []

    @property
    def token_count(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def search(self, query: str) -> None:
        if self._bm25_only:
            results = self._searcher.bm25_search(query, limit=self._limit)
        elif self._hybrid:
            results = self._searcher.hybrid_search(
                query, limit=self._limit, collection=self._collection, tags=self._tags)
        else:
            results = self._searcher.vector_search(
                query, limit=self._limit, collection=self._collection, tags=self._tags)

        self.context = []
        for r in results:
            self.context.append({
                "key": r.doc_id,
                "title": r.title,
                "text": r.snippet,
                "score": r.score,
            })

        # Build system prompt with paper context
        context_parts = []
        for i, doc in enumerate(self.context, 1):
            context_parts.append(f"[{i}] {doc['key']} - {doc['title']}\n{doc['text']}")
        context_text = "\n\n".join(context_parts) if context_parts else "No papers found."

        self.messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nPapers:\n\n{context_text}"},
        ]

    def ask(self, question: str) -> Iterator[str]:
        self.messages.append({"role": "user", "content": question})
        full_response = []
        for token in self._llm.stream_messages(self.messages):
            full_response.append(token)
            yield token
        self.messages.append({"role": "assistant", "content": "".join(full_response)})
