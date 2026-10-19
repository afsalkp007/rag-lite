"""
LLM answer generation providers for RAG Lite.
Enforces grounded generation strictly derived from retrieved context chunks.
"""

from abc import ABC, abstractmethod
import json
import os
import urllib.request
from typing import List, Optional

from .models import RetrievalResult

SYSTEM_GROUNDING_PROMPT = """You are a precise, factual enterprise technical assistant.
Answer the user's question relying STRICTLY and EXCLUSIVELY on the provided context excerpts.

RULES:
1. Do not use outside knowledge or speculate.
2. If the context does not contain enough information to answer the question with certainty, respond exactly with:
   "I cannot answer this question based on the provided knowledge documents."
3. When stating facts, cite the source document name in parentheses (e.g. '(Source: Stripe Billing Guide)').
"""


class BaseLLMGenerator(ABC):
    """Abstract interface for RAG answer generation."""

    @abstractmethod
    def generate_grounded_answer(self, query: str, contexts: List[RetrievalResult]) -> str:
        """Generate an answer strictly grounded in the retrieved contexts."""
        pass


class GeminiGenerator(BaseLLMGenerator):
    """Google Gemini 2.5 Flash generation backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", model)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiGenerator.")

    def generate_grounded_answer(self, query: str, contexts: List[RetrievalResult]) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        context_str = "\n\n".join([
            f"--- Context {idx + 1} (From '{c.chunk.document_title}') ---\n{c.chunk.text}"
            for idx, c in enumerate(contexts)
        ])
        user_prompt = f"Context Information:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_GROUNDING_PROMPT}]},
            "generationConfig": {"temperature": 0.0}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()


class OpenAIGenerator(BaseLLMGenerator):
    """OpenAI generation backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", model)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerator.")

    def generate_grounded_answer(self, query: str, contexts: List[RetrievalResult]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        context_str = "\n\n".join([
            f"--- Context {idx + 1} ('{c.chunk.document_title}') ---\n{c.chunk.text}"
            for idx, c in enumerate(contexts)
        ])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_GROUNDING_PROMPT},
                {"role": "user", "content": f"Context Information:\n{context_str}\n\nQuestion: {query}"}
            ],
            "temperature": 0.0
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


class MockGenerator(BaseLLMGenerator):
    """
    Deterministic generation backend for offline testing, CI/CD,
    and instant demo verification without API keys.
    """

    def generate_grounded_answer(self, query: str, contexts: List[RetrievalResult]) -> str:
        if not contexts:
            return "I cannot answer this question based on the provided knowledge documents."

        top_chunk = contexts[0].chunk
        doc_title = top_chunk.document_title

        # Synthesize a grounded answer based on the best matched chunk
        lines = top_chunk.text.split("\n")
        relevant_sentences = [l.strip() for l in lines if len(l.strip()) > 30][:3]
        synthesis = " ".join(relevant_sentences) if relevant_sentences else top_chunk.text[:200]

        return f"Based on the official documentation ({doc_title}): {synthesis}"
