"""
Domain models for RAG Lite using Pydantic v2.
Type-safe representations of documents, chunks, similarity retrieval, and grounded answers.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A raw document ingested into the knowledge base."""
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title or filename")
    content: str = Field(..., description="Full text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")


class DocumentChunk(BaseModel):
    """A segment of text created during document chunking."""
    chunk_id: str = Field(..., description="Deterministic chunk identifier (e.g. doc-1_chunk-0)")
    document_id: str = Field(..., description="Parent document identifier")
    document_title: str = Field(..., description="Parent document title for citations")
    text: str = Field(..., description="The chunk text content")
    chunk_index: int = Field(..., ge=0, description="Sequential index of the chunk in the document")
    start_char: int = Field(..., ge=0, description="Start character offset in original document")
    end_char: int = Field(..., ge=0, description="End character offset in original document")
    embedding: Optional[List[float]] = Field(default=None, description="Dense vector embedding")


class RetrievalResult(BaseModel):
    """A single chunk retrieved by similarity search."""
    chunk: DocumentChunk = Field(..., description="The matched document chunk")
    score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score (0.0 to 1.0)")


class RAGResponse(BaseModel):
    """The synthesized answer returned to the user."""
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded answer or abstention message")
    sources: List[RetrievalResult] = Field(default_factory=list, description="Retrieved source chunks used")
    is_abstained: bool = Field(default=False, description="True if confidence fell below abstention threshold")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Top retrieval similarity score")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end pipeline latency in milliseconds")
