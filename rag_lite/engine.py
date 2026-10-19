"""
RAGPipeline orchestrator.
Manages ingestion, chunking, embedding, vector retrieval, and grounded answer synthesis.
"""

from pathlib import Path
import time
from typing import List, Optional, Union

from .models import Document, DocumentChunk, RetrievalResult, RAGResponse
from .chunker import RecursiveCharacterChunker
from .embeddings import BaseEmbeddingProvider, LocalSemanticEmbeddingProvider
from .vector_store import InMemoryVectorStore
from .providers import BaseLLMGenerator, GeminiGenerator, OpenAIGenerator, MockGenerator
import os


class RAGPipeline:
    """
    End-to-end RAG system with customizable chunking, vector indexing,
    and confidence-based abstention guardrails.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        generator: Optional[BaseLLMGenerator] = None,
        default_score_threshold: float = 0.40
    ):
        self.chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_provider = embedding_provider or LocalSemanticEmbeddingProvider()
        self.vector_store = InMemoryVectorStore(embedding_provider=self.embedding_provider)
        self.generator = generator or self._auto_detect_generator()
        self.default_score_threshold = default_score_threshold

    def _auto_detect_generator(self) -> BaseLLMGenerator:
        prov = os.getenv("LLM_PROVIDER", "").lower()
        if prov == "gemini" and os.getenv("GEMINI_API_KEY"):
            return GeminiGenerator()
        elif prov == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIGenerator()
        elif os.getenv("GEMINI_API_KEY"):
            return GeminiGenerator()
        elif os.getenv("OPENAI_API_KEY"):
            return OpenAIGenerator()
        return MockGenerator()

    def ingest_text(self, title: str, text: str, doc_id: Optional[str] = None) -> int:
        """Ingest raw string content as a Document."""
        document = Document(
            id=doc_id or f"doc_{int(time.time() * 1000)}",
            title=title,
            content=text
        )
        chunks = self.chunker.chunk_document(document)
        return self.vector_store.add_chunks(chunks)

    def ingest_file(self, file_path: Union[str, Path]) -> int:
        """Read and index a single Markdown or text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ").title()
        return self.ingest_text(title=title, text=text, doc_id=path.stem)

    def ingest_directory(self, dir_path: Union[str, Path]) -> int:
        """Recursively index all .md and .txt files in a directory."""
        folder = Path(dir_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        total_chunks = 0
        for ext in ["*.md", "*.txt"]:
            for f in folder.glob(ext):
                total_chunks += self.ingest_file(f)

        return total_chunks

    def query(
        self,
        question: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> RAGResponse:
        """
        Execute grounded RAG query.
        If similarity is below score_threshold, abstains gracefully to prevent hallucination.
        """
        start_time = time.perf_counter()
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold

        # 1. Similarity Retrieval
        contexts: List[RetrievalResult] = self.vector_store.search(
            query=question,
            top_k=top_k,
            score_threshold=threshold
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        top_score = contexts[0].score if contexts else 0.0

        # 2. Abstention Guardrail: If no context meets threshold, refuse to answer
        if not contexts:
            return RAGResponse(
                query=question,
                answer=(
                    "I cannot answer this question based on the provided knowledge documents. "
                    "(Abstained: No matching documentation exceeded the confidence threshold)."
                ),
                sources=[],
                is_abstained=True,
                confidence_score=0.0,
                latency_ms=elapsed_ms
            )

        # 3. Grounded Answer Synthesis
        answer = self.generator.generate_grounded_answer(question, contexts)
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=contexts,
            is_abstained=False,
            confidence_score=top_score,
            latency_ms=total_latency_ms
        )
