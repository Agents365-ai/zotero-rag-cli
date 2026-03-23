import pytest

from rak.store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=tmp_path / "chroma", dimension=384)


def test_add_and_search(store):
    store.add(
        ids=["A1", "A2"],
        embeddings=[[1.0] + [0.0] * 383, [0.0] + [1.0] + [0.0] * 382],
        documents=["paper about RNA", "paper about DNA"],
        metadatas=[{"title": "RNA paper"}, {"title": "DNA paper"}],
    )
    results = store.search(query_embedding=[1.0] + [0.0] * 383, limit=1)
    assert len(results) == 1
    assert results[0]["id"] == "A1"


def test_search_returns_score(store):
    store.add(
        ids=["X1"],
        embeddings=[[1.0] + [0.0] * 383],
        documents=["test"],
        metadatas=[{"title": "test"}],
    )
    results = store.search(query_embedding=[1.0] + [0.0] * 383, limit=1)
    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_count(store):
    assert store.count() == 0
    store.add(
        ids=["Z1"],
        embeddings=[[0.5] * 384],
        documents=["doc"],
        metadatas=[{"title": "doc"}],
    )
    assert store.count() == 1


def test_clear(store):
    store.add(
        ids=["Z1"],
        embeddings=[[0.5] * 384],
        documents=["doc"],
        metadatas=[{"title": "doc"}],
    )
    store.clear()
    assert store.count() == 0


def test_has(store):
    assert not store.has("Z1")
    store.add(
        ids=["Z1"],
        embeddings=[[0.5] * 384],
        documents=["doc"],
        metadatas=[{"title": "doc"}],
    )
    assert store.has("Z1")


def test_delete(store):
    store.add(
        ids=["A1", "A2"],
        embeddings=[[1.0] + [0.0] * 383, [0.0] + [1.0] + [0.0] * 382],
        documents=["paper one", "paper two"],
        metadatas=[{"title": "one"}, {"title": "two"}],
    )
    assert store.count() == 2
    store.delete(["A1"])
    assert store.count() == 1
    assert not store.has("A1")
    assert store.has("A2")


def test_search_with_where_filter(store):
    store.add(
        ids=["A1", "A2"],
        embeddings=[[1.0] + [0.0] * 383, [0.9] + [0.1] + [0.0] * 382],
        documents=["paper about RNA", "paper about DNA"],
        metadatas=[
            {"title": "RNA paper", "collections": ["Biology", "Genomics"], "tags": ["RNA", "seq"]},
            {"title": "DNA paper", "collections": ["Chemistry"], "tags": ["DNA"]},
        ],
    )
    results = store.search(
        query_embedding=[1.0] + [0.0] * 383,
        limit=10,
        where={"collections": {"$contains": "Biology"}},
    )
    assert len(results) == 1
    assert results[0]["id"] == "A1"


def test_get_embedding(store):
    emb = [1.0] + [0.0] * 383
    store.add(ids=["E1"], embeddings=[emb], documents=["doc"], metadatas=[{"title": "t"}])
    result = store.get_embedding("E1")
    assert result is not None
    assert len(result) == 384
    assert result[0] == pytest.approx(1.0, abs=0.01)


def test_get_embedding_missing(store):
    result = store.get_embedding("NONEXIST")
    assert result is None
