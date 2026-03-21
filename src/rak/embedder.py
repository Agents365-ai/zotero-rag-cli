from __future__ import annotations

from sentence_transformers import SentenceTransformer

from rak.config import DEFAULT_MODEL


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
