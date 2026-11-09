from hybrid_rag.infrastructure.qdrant_store import QdrantVectorStore


def main() -> None:
    store = QdrantVectorStore(
        storage_path="data/qdrant_storage",
        collection_name="hybrid_rag_chunks",
        dimensions=384,
    )

    try:
        collection = store.client.get_collection(
            store.collection_name
        )

        print("Qdrant collection initialized successfully")
        print(f"Collection: {store.collection_name}")
        print(f"Dimensions: {store.dimensions}")
        print(f"Distance: {collection.config.params.vectors.distance}")
        print(f"Stored points: {collection.points_count}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
