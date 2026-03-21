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
