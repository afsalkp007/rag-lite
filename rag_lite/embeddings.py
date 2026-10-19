"""
Embedding providers for RAG Lite.
Includes pure NumPy/Python local semantic hashing for zero-dependency offline speed,
as well as live cloud embedding APIs (Gemini and OpenAI).
"""

from abc import ABC, abstractmethod
import hashlib
import json
import math
import os
import re
import urllib.request
from typing import List, Optional
import numpy as np


class BaseEmbeddingProvider(ABC):
    """Abstract interface for generating dense vector embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a normalized vector."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        return [self.embed_text(t) for t in texts]


STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "our", "we", "my", "your", "can", "how"
}


class LocalSemanticEmbeddingProvider(BaseEmbeddingProvider):
    """
    Fast, zero-dependency local semantic vectorizer using subword n-gram hashing,
    stopword filtering, and bigram feature projections. Runs 100% offline.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        clean_text = text.lower().strip()
        all_tokens = re.findall(r"\b[a-z0-9_]{2,}\b", clean_text)
        if not all_tokens:
            return vec.tolist()

        # Filter out stopwords for core semantic representation
        content_tokens = [t for t in all_tokens if t not in STOPWORDS]
        if not content_tokens:
            content_tokens = all_tokens

        features = []
        # Word features (weight = 2.0)
        for t in content_tokens:
            features.append((t, 2.0))
            if len(t) >= 4:
                features.append((t[:4], 1.0))
                features.append((t[-3:], 1.0))

        # Bigram phrase features (weight = 2.5)
        for i in range(len(content_tokens) - 1):
            bigram = f"{content_tokens[i]}_{content_tokens[i+1]}"
            features.append((bigram, 2.5))

        for feature_text, weight in features:
            h = int(hashlib.md5(feature_text.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign * weight

        # L2 Normalization (so dot product equals cosine similarity)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Embeddings using Google Gemini text-embedding-004 API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiEmbeddingProvider.")

    def embed_text(self, text: str) -> List[float]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
            f"?key={self.api_key}"
        )
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text[:2000]}]}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]["values"]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embeddings using OpenAI text-embedding-3-small API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")

    def embed_text(self, text: str) -> List[float]:
        url = "https://api.openai.com/v1/embeddings"
        payload = {"input": text[:2000], "model": self.model}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]
