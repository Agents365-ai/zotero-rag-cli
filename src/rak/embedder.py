from __future__ import annotations

from rak.config import DEFAULT_MODEL


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 provider: str = "local",
                 base_url: str = "http://localhost:11434/v1",
                 api_key: str = "not-needed") -> None:
        self._model_name = model_name
        self._provider = provider
        if provider == "api":
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            self._dimension: int | None = None
        else:
            try:
                import logging, os
                for name in ("sentence_transformers", "transformers", "safetensors"):
                    logging.getLogger(name).setLevel(logging.WARNING)
                os.environ["SAFETENSORS_FAST_GPU"] = "1"
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, trust_remote_code=True)
            except Exception as exc:
                from rak.errors import ModelDownloadError
                raise ModelDownloadError(model_name, str(exc)) from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._provider == "api":
            if self._dimension is None:
                vec = self.embed("dimension probe")
                self._dimension = len(vec)
            return self._dimension
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        if self._provider == "api":
            resp = self._client.embeddings.create(input=[text], model=self._model_name)
            return resp.data[0].embedding
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._provider == "api":
            resp = self._client.embeddings.create(input=texts, model=self._model_name)
            return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
