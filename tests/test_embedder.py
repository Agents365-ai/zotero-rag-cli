import pytest

from rak.embedder import Embedder


@pytest.fixture
def embedder():
    return Embedder(model_name="all-MiniLM-L6-v2")


@pytest.mark.network
def test_embed_single(embedder):
    vec = embedder.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.network
def test_embed_batch(embedder):
    texts = ["paper about RNA", "deep learning methods", "cell biology"]
    vecs = embedder.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.network
def test_embed_empty_string(embedder):
    vec = embedder.embed("")
    assert len(vec) == 384


@pytest.mark.network
def test_model_name(embedder):
    assert embedder.model_name == "all-MiniLM-L6-v2"


@pytest.mark.network
def test_dimension(embedder):
    assert embedder.dimension == 384


# --- API provider tests (no network needed) ---

from unittest.mock import patch, MagicMock


def test_api_provider_embed():
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock()]
    mock_resp.data[0].embedding = [0.1, 0.2, 0.3]

    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.return_value = mock_resp
        emb = Embedder(model_name="text-embedding-3-small", provider="api",
                       base_url="http://localhost:11434/v1", api_key="test-key")
        vec = emb.embed("hello world")

    assert vec == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        input=["hello world"], model="text-embedding-3-small"
    )


def test_api_provider_embed_batch():
    mock_resp = MagicMock()
    d0 = MagicMock()
    d0.embedding = [0.1, 0.2]
    d0.index = 0
    d1 = MagicMock()
    d1.embedding = [0.3, 0.4]
    d1.index = 1
    mock_resp.data = [d1, d0]  # intentionally out of order

    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.return_value = mock_resp
        emb = Embedder(model_name="text-embedding-3-small", provider="api")
        vecs = emb.embed_batch(["text1", "text2"])

    assert vecs == [[0.1, 0.2], [0.3, 0.4]]  # sorted by index


def test_api_provider_dimension():
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock()]
    mock_resp.data[0].embedding = [0.0] * 1536

    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.return_value = mock_resp
        emb = Embedder(model_name="text-embedding-3-small", provider="api")
        dim = emb.dimension

    assert dim == 1536


def test_api_provider_model_name():
    with patch("openai.OpenAI"):
        emb = Embedder(model_name="custom-model", provider="api")
    assert emb.model_name == "custom-model"
