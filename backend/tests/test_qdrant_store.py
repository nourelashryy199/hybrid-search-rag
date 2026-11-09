import pytest

from hybrid_rag.domain.document import DocumentChunk
from hybrid_rag.domain.embedding import EmbeddedChunk
from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore


@pytest.fixture
def store(tmp_path):
    instance = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="test_chunks",
        dimensions=3,
    )

    yield instance

    instance.close()


def make_chunk(chunk_id: str = "chunk-001") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        text="Employees receive 25 days of annual leave.",
        chunk_index=0,
        page_number=1,
    )


def make_embedding(
    chunk_id: str = "chunk-001",
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        vector=vector,
    )


def test_add_stores_vector_and_payload(store):
    chunk = make_chunk()
    embedding = make_embedding()

    store.add([chunk], [embedding])

    points = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[store._point_id(chunk.chunk_id)],
        with_payload=True,
        with_vectors=True,
    )

    assert len(points) == 1

    point = points[0]

    assert point.payload["chunk_id"] == chunk.chunk_id
    assert point.payload["document_id"] == chunk.document_id
    assert point.payload["text"] == chunk.text
    assert point.payload["chunk_index"] == chunk.chunk_index
    assert point.payload["page_number"] == chunk.page_number

    assert point.vector == pytest.approx([1.0, 0.0, 0.0])


def test_add_same_chunk_twice_does_not_duplicate(store):
    chunk = make_chunk()
    embedding = make_embedding()

    store.add([chunk], [embedding])
    store.add([chunk], [embedding])

    collection = store.client.get_collection(store.collection_name)

    assert collection.points_count == 1


def test_add_rejects_mismatched_counts(store):
    with pytest.raises(ValueError, match="number of chunks"):
        store.add([make_chunk()], [])


def test_add_rejects_mismatched_ids(store):
    with pytest.raises(ValueError, match="IDs must match"):
        store.add(
            [make_chunk("chunk-001")],
            [make_embedding("chunk-002")],
        )


def test_add_rejects_wrong_dimensions(store):
    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        store.add(
            [make_chunk()],
            [make_embedding(vector=(1.0, 0.0))],
        )


def test_add_rejects_duplicate_chunk_ids(store):
    chunk = make_chunk()
    embedding = make_embedding()

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        store.add(
            [chunk, chunk],
            [embedding, embedding],
        )


def test_add_empty_batch_is_noop(store):
    store.add([], [])

    collection = store.client.get_collection(store.collection_name)

    assert collection.points_count == 0


def test_add_validates_entire_batch_before_upsert(store):
    chunks = [
        make_chunk("chunk-001"),
        make_chunk("chunk-002"),
    ]

    embeddings = [
        make_embedding("chunk-001"),
        make_embedding("chunk-002", vector=(1.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        store.add(chunks, embeddings)

    collection = store.client.get_collection(store.collection_name)

    assert collection.points_count == 0

def test_delete_removes_existing_vector(store):
    chunk = make_chunk()
    embedding = make_embedding()

    store.add([chunk], [embedding])

    store.delete([chunk.chunk_id])

    points = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[store._point_id(chunk.chunk_id)],
    )

    assert points == []


def test_delete_removes_only_requested_vectors(store):
    chunk_a = make_chunk("chunk-001")
    chunk_b = make_chunk("chunk-002")

    embedding_a = make_embedding("chunk-001")
    embedding_b = make_embedding(
        "chunk-002",
        vector=(0.0, 1.0, 0.0),
    )

    store.add(
        [chunk_a, chunk_b],
        [embedding_a, embedding_b],
    )

    store.delete(["chunk-001"])

    remaining = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[
            store._point_id("chunk-001"),
            store._point_id("chunk-002"),
        ],
    )

    assert len(remaining) == 1
    assert remaining[0].payload["chunk_id"] == "chunk-002"


def test_delete_missing_chunk_is_harmless(store):
    store.delete(["nonexistent-chunk"])

    collection = store.client.get_collection(store.collection_name)

    assert collection.points_count == 0


def test_delete_empty_list_is_noop(store):
    store.delete([])

    collection = store.client.get_collection(store.collection_name)

    assert collection.points_count == 0


def test_delete_duplicate_ids(store):
    chunk = make_chunk()
    embedding = make_embedding()

    store.add([chunk], [embedding])

    store.delete([
        chunk.chunk_id,
        chunk.chunk_id,
        chunk.chunk_id,
    ])

    collection = store.client.get_collection(store.collection_name)

    assert collection.points_count == 0
def test_search_returns_most_similar_chunk(store):
    chunk_a = make_chunk("chunk-001")
    chunk_b = make_chunk("chunk-002")

    store.add(
        [chunk_a, chunk_b],
        [
            make_embedding("chunk-001", (1.0, 0.0, 0.0)),
            make_embedding("chunk-002", (0.0, 1.0, 0.0)),
        ],
    )

    results = store.search(
        query_vector=[0.9, 0.1, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-001"
    assert results[1].chunk.chunk_id == "chunk-002"
    assert results[0].score > results[1].score


def test_search_reconstructs_document_metadata(store):
    chunk = make_chunk()

    store.add(
        [chunk],
        [make_embedding()],
    )

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk == chunk
    assert results[0].score == pytest.approx(1.0)


def test_search_respects_top_k(store):
    chunks = [
        make_chunk("chunk-001"),
        make_chunk("chunk-002"),
        make_chunk("chunk-003"),
    ]

    embeddings = [
        make_embedding("chunk-001", (1.0, 0.0, 0.0)),
        make_embedding("chunk-002", (0.0, 1.0, 0.0)),
        make_embedding("chunk-003", (0.0, 0.0, 1.0)),
    ]

    store.add(chunks, embeddings)

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_search_empty_collection_returns_empty_list(store):
    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert results == []


def test_search_rejects_invalid_top_k(store):
    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search([1.0, 0.0, 0.0], top_k=0)


def test_search_rejects_wrong_dimensions(store):
    with pytest.raises(ValueError, match="Expected 3 query dimensions"):
        store.search([1.0, 0.0], top_k=5)


def test_search_rejects_zero_vector(store):
    with pytest.raises(ValueError, match="cannot be zero"):
        store.search([0.0, 0.0, 0.0], top_k=5)


def test_search_rejects_non_finite_values(store):
    with pytest.raises(ValueError, match="finite values"):
        store.search([float("nan"), 0.0, 1.0], top_k=5)

    with pytest.raises(ValueError, match="finite values"):
        store.search([float("inf"), 0.0, 1.0], top_k=5)

def test_vectors_persist_after_reopening_database(tmp_path):
    storage_path = tmp_path / "persistent_qdrant"
    collection_name = "test_persistence"

    chunk = make_chunk("persistent-chunk-001")
    embedding = make_embedding(
        "persistent-chunk-001",
        vector=(1.0, 0.0, 0.0),
    )

    # First process lifecycle: create and populate the database.
    first_store = QdrantVectorStore(
        storage_path=storage_path,
        collection_name=collection_name,
        dimensions=3,
    )

    try:
        first_store.add([chunk], [embedding])
    finally:
        first_store.close()

    # Second lifecycle: reopen the existing database.
    second_store = QdrantVectorStore(
        storage_path=storage_path,
        collection_name=collection_name,
        dimensions=3,
    )

    try:
        results = second_store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk == chunk
        assert results[0].score == pytest.approx(1.0)

        collection = second_store.client.get_collection(
            collection_name
        )

        assert collection.points_count == 1
    finally:
        second_store.close()

def test_reopening_collection_rejects_dimension_mismatch(tmp_path):
    storage_path = tmp_path / "qdrant"
    collection_name = "test_dimensions"

    first_store = QdrantVectorStore(
        storage_path=storage_path,
        collection_name=collection_name,
        dimensions=3,
    )
    first_store.close()

    with pytest.raises(ValueError, match="dimension mismatch"):
        QdrantVectorStore(
            storage_path=storage_path,
            collection_name=collection_name,
            dimensions=4,
        )
