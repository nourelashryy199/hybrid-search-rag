
import pytest

from hybrid_rag.application.chunking import (
    ChunkingConfig,
    TextChunker,
)
from hybrid_rag.domain.document import DocumentPage


def make_page(
    text: str,
    page_number: int = 1,
) -> DocumentPage:
    return DocumentPage(
        document_id="doc-001",
        page_number=page_number,
        text=text,
    )


def test_short_page_creates_one_chunk():
    chunker = TextChunker(
        ChunkingConfig(chunk_size=100, chunk_overlap=20)
    )

    chunks = chunker.chunk([make_page("Short document.")])

    assert len(chunks) == 1
    assert chunks[0].text == "Short document."


def test_long_page_creates_overlapping_chunks():
    chunker = TextChunker(
        ChunkingConfig(chunk_size=10, chunk_overlap=3)
    )

    chunks = chunker.chunk([make_page("abcdefghijklmnopqrstuvwxyz")])

    assert len(chunks) == 4
    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text == "hijklmnopq"
    assert chunks[0].text[-3:] == chunks[1].text[:3]


def test_chunker_preserves_page_numbers():
    chunker = TextChunker(
        ChunkingConfig(chunk_size=100, chunk_overlap=10)
    )

    pages = [
        make_page("First page.", page_number=1),
        make_page("Third page.", page_number=3),
    ]

    chunks = chunker.chunk(pages)

    assert [c.page_number for c in chunks] == [1, 3]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_chunk_ids_are_deterministic():
    chunker = TextChunker()

    pages = [make_page("Some example content.")]

    first = chunker.chunk(pages)
    second = chunker.chunk(pages)

    assert first[0].chunk_id == second[0].chunk_id


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_empty_page_list():
    chunker = TextChunker()

    assert chunker.chunk([]) == []
