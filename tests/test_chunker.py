import pytest
from rag_lite.chunker import RecursiveCharacterChunker
from rag_lite.models import Document


def test_chunker_basic():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    doc = Document(
        id="test_doc",
        title="Test Document",
        content=(
            "First long paragraph explaining architecture concepts in depth.\n\n"
            "Second long paragraph covering performance benchmarks and latency considerations.\n\n"
            "Third long paragraph summarizing conclusions and deployment recommendations."
        )
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert chunks[0].document_id == "test_doc"
    assert chunks[0].start_char >= 0
    assert chunks[0].end_char > chunks[0].start_char


def test_chunker_empty_document():
    chunker = RecursiveCharacterChunker()
    doc = Document(id="empty", title="Empty", content="   ")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=120)
