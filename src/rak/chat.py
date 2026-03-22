from __future__ import annotations

import sys
from collections.abc import Iterator

from rak.llm import LLMClient, SYSTEM_PROMPT
from rak.searcher import Searcher


class ChatSession:
    def __init__(self, searcher: Searcher, llm: LLMClient, limit: int = 5,
                 collection: str | None = None, tags: list[str] | None = None,
                 hybrid: bool = False) -> None:
        self._searcher = searcher
        self._llm = llm
        self._limit = limit
        self._collection = collection
        self._tags = tags
        self._hybrid = hybrid
        self.context: list[dict] = []
        self.messages: list[dict] = []

    def search(self, query: str, vector_store=None) -> None:
        if self._hybrid:
            results = self._searcher.hybrid_search(
                query, limit=self._limit, collection=self._collection, tags=self._tags)
        else:
            results = self._searcher.vector_search(
                query, limit=self._limit, collection=self._collection, tags=self._tags)

        self.context = []
        for r in results:
            doc_text = ""
            if vector_store:
                doc_data = vector_store._collection.get(ids=[r.doc_id], include=["documents"])
                doc_text = doc_data["documents"][0] if doc_data["documents"] else ""
            self.context.append({
                "key": r.doc_id,
                "title": r.title,
                "text": doc_text,
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
