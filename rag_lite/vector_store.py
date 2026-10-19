"""
In-memory dense vector store with vectorized cosine similarity and abstention thresholds.
Follows clean code principles with zero C++ compilation dependencies.
"""

from typing import List, Optional
import numpy as np

from .models import DocumentChunk, RetrievalResult
from .embeddings import BaseEmbeddingProvider, LocalSemanticEmbeddingProvider


class InMemoryVectorStore:
    """
    Lightweight, vectorized in-memory vector database.
    Performs fast cosine similarity ranking using NumPy matrix operations.
    """

    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or LocalSemanticEmbeddingProvider()
        self._chunks: List[DocumentChunk] = []
        self._matrix: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed and index a list of DocumentChunks."""
        if not chunks:
            return 0

        # Check for missing embeddings and batch compute
        missing_texts = [c.text for c in chunks if c.embedding is None]
        if missing_texts:
            new_embeddings = self.embedding_provider.embed_batch(missing_texts)
            embed_idx = 0
            for c in chunks:
                if c.embedding is None:
                    c.embedding = new_embeddings[embed_idx]
                    embed_idx += 1

        self._chunks.extend(chunks)

        # Rebuild dense matrix
        vectors = [c.embedding for c in self._chunks if c.embedding is not None]
        if vectors:
            self._matrix = np.array(vectors, dtype=np.float32)
            # Ensure row normalization
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = self._matrix / norms

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.45
    ) -> List[RetrievalResult]:
        """
        Compute cosine similarity of query against all indexed chunks.
        Returns top_k matches that meet or exceed score_threshold.
        """
        if self._matrix is None or len(self._chunks) == 0:
            return []

        # Embed query vector
        query_vec = np.array(self.embedding_provider.embed_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm
        else:
            return []

        # Vectorized dot product (cosine similarity)
        scores = np.dot(self._matrix, query_vec)

        # Calibrate sparse hash vectors to standard 0.0 - 1.0 confidence distribution
        if isinstance(self.embedding_provider, LocalSemanticEmbeddingProvider):
            scores = np.clip(scores * 3.5, 0.0, 1.0)

        # Filter by threshold and rank
        matched_indices = np.where(scores >= score_threshold)[0]
        if len(matched_indices) == 0:
            return []

        # Sort descending by score
        sorted_order = matched_indices[np.argsort(-scores[matched_indices])]
        top_indices = sorted_order[:top_k]

        results = [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=float(round(scores[idx], 4))
            )
            for idx in top_indices
        ]
        return results

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return len(self._chunks)

    def list_documents(self) -> List[str]:
        """Return distinct document titles indexed."""
        return list(dict.fromkeys(c.document_title for c in self._chunks))

    def clear(self) -> None:
        """Reset the vector index."""
        self._chunks.clear()
        self._matrix = None
