import pytest
from pydantic import ValidationError
from rag_lite.models import Document, DocumentChunk, RetrievalResult, RAGResponse


def test_document_model():
    doc = Document(id="doc-1", title="Title", content="Content")
    assert doc.id == "doc-1"
    assert doc.title == "Title"


def test_document_chunk_model():
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        document_title="Title",
        text="Sample chunk text",
        chunk_index=0,
        start_char=0,
        end_char=17
    )
    assert chunk.chunk_id == "c1"
    assert chunk.end_char == 17


def test_rag_response_bounds():
    with pytest.raises(ValidationError):
        RAGResponse(
            query="Query",
            answer="Answer",
            confidence_score=1.5,  # Out of bounds (> 1.0)
            latency_ms=10.0
        )
