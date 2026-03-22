from __future__ import annotations

from pathlib import Path

import chromadb


class VectorStore:
    COLLECTION_NAME = "rak_papers"

    def __init__(self, persist_dir: Path, dimension: int = 384) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._dimension = dimension

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def search(self, query_embedding: list[float], limit: int = 10, where: dict | None = None) -> list[dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],
            })
        return items

    def count(self) -> int:
        return self._collection.count()

    def get(self, ids: list[str], include: list[str] | None = None) -> dict:
        """Retrieve documents/metadatas by IDs."""
        return self._collection.get(ids=ids, include=include or ["documents"])

    def get_by_metadata(self, where: dict) -> dict:
        """Query documents by metadata filter."""
        return self._collection.get(where=where, include=[])

    def has(self, doc_id: str) -> bool:
        result = self._collection.get(ids=[doc_id], include=[])
        return len(result["ids"]) > 0

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
        )
