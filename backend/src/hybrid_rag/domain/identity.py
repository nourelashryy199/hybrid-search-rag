
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    content_hash: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")

        if len(self.content_hash) != 64:
            raise ValueError(
                "content_hash must be a SHA-256 hex digest"
            )

        try:
            bytes.fromhex(self.content_hash)
        except ValueError as exc:
            raise ValueError(
                "content_hash must be hexadecimal"
            ) from exc


def calculate_file_hash(
    file_path: str | Path,
) -> str:
    path = Path(file_path)

    digest = sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def create_document_identity(
    file_path: str | Path,
    document_id: str,
) -> DocumentIdentity:
    return DocumentIdentity(
        document_id=document_id,
        content_hash=calculate_file_hash(file_path),
    )
