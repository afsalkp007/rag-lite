"""
Text chunking engine for RAG Lite.
Implements Recursive Character Chunking with configurable overlap to preserve semantic coherence.
"""

from typing import List
from .models import Document, DocumentChunk


class RecursiveCharacterChunker:
    """
    Splits text recursively across natural semantic boundaries:
    Paragraphs (\\n\\n) -> Lines (\\n) -> Sentences (. ) -> Words ( )
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a Document into indexed DocumentChunks with overlap."""
        text = document.content
        if not text.strip():
            return []

        chunks: List[DocumentChunk] = []
        raw_splits = self._split_text(text, self.separators)

        # Merge splits into windows of size chunk_size with chunk_overlap
        current_chunk_pieces: List[str] = []
        current_length = 0
        chunk_idx = 0
        current_start_offset = 0

        for piece in raw_splits:
            piece_len = len(piece)
            if current_length + piece_len > self.chunk_size and current_chunk_pieces:
                # Flush current chunk
                chunk_text = "".join(current_chunk_pieces).strip()
                if chunk_text:
                    end_offset = current_start_offset + len(chunk_text)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{document.id}_chunk-{chunk_idx}",
                        document_id=document.id,
                        document_title=document.title,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=current_start_offset,
                        end_char=end_offset
                    ))
                    chunk_idx += 1

                # Calculate overlap retention
                overlap_pieces: List[str] = []
                overlap_len = 0
                for prev_piece in reversed(current_chunk_pieces):
                    if overlap_len + len(prev_piece) <= self.chunk_overlap:
                        overlap_pieces.insert(0, prev_piece)
                        overlap_len += len(prev_piece)
                    else:
                        break

                current_chunk_pieces = list(overlap_pieces)
                current_length = overlap_len
                current_start_offset = max(0, end_offset - overlap_len)

            current_chunk_pieces.append(piece)
            current_length += piece_len

        # Flush any remaining text
        if current_chunk_pieces:
            chunk_text = "".join(current_chunk_pieces).strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    chunk_id=f"{document.id}_chunk-{chunk_idx}",
                    document_id=document.id,
                    document_title=document.title,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    start_char=current_start_offset,
                    end_char=current_start_offset + len(chunk_text)
                ))

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators."""
        if not separators:
            return list(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        final_pieces: List[str] = []

        for i, split in enumerate(splits):
            piece = split if i == len(splits) - 1 else split + sep
            if len(piece) > self.chunk_size and remaining_seps:
                final_pieces.extend(self._split_text(piece, remaining_seps))
            else:
                final_pieces.append(piece)

        return final_pieces
