import sqlite3
from pathlib import Path

from hybrid_rag.domain.identity import DocumentIdentity
from hybrid_rag.domain.indexing import IndexedDocument


class SQLiteDocumentRepository:
    """Persist active document versions and chunk IDs using SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (document_id, chunk_id),
                    UNIQUE (document_id, position),
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                )
                """
            )

    def get(self, document_id: str) -> IndexedDocument | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, content_hash
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if row is None:
                return None

            chunk_rows = connection.execute(
                """
                SELECT chunk_id
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()

        identity = DocumentIdentity(
            document_id=row[0],
            content_hash=row[1],
        )

        return IndexedDocument(
            identity=identity,
            chunk_ids=tuple(
                chunk_row[0]
                for chunk_row in chunk_rows
            ),
        )

    def save(self, document: IndexedDocument) -> None:
        """Atomically replace a document's active version metadata."""

        document_id = document.identity.document_id

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (document_id, content_hash)
                VALUES (?, ?)
                ON CONFLICT(document_id)
                DO UPDATE SET content_hash = excluded.content_hash
                """,
                (
                    document_id,
                    document.identity.content_hash,
                ),
            )

            connection.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = ?
                """,
                (document_id,),
            )

            connection.executemany(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_id,
                    position
                )
                VALUES (?, ?, ?)
                """,
                [
                    (document_id, chunk_id, position)
                    for position, chunk_id in enumerate(
                        document.chunk_ids
                    )
                ],
            )

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            )

    def list_all(self) -> list[IndexedDocument]:
        with self._connect() as connection:
            document_ids = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT document_id
                    FROM documents
                    ORDER BY document_id
                    """
                ).fetchall()
            ]

        return [
            document
            for document_id in document_ids
            if (document := self.get(document_id)) is not None
        ]
