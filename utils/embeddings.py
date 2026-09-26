"""Sentence-transformer retrieval and cosine-similarity utilities."""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Dict, List, Protocol, Sequence


class TextEncoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ): ...


def normalize_text(text: str) -> str:
    """Return lowercase alphanumeric tokens for lexical checks."""
    return " ".join(re.findall(r"\b\w+\b", text.lower()))


class SentenceTransformerIndex:
    """In-memory dense index that encodes corpus documents once per instance."""

    def __init__(
        self,
        documents: List[Dict],
        model_name: str | None = None,
        model: TextEncoder | None = None,
    ):
        self.documents = list(documents)
        self.model_name = model_name or os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = model if model is not None else self._load_model(self.model_name)

        self.document_texts = [self.document_text(doc) for doc in self.documents]
        unique_texts = list(dict.fromkeys(self.document_texts))
        encoded = self._encode(unique_texts) if unique_texts else []
        self._embedding_cache = dict(zip(unique_texts, encoded))
        self.document_embeddings = [
            self._embedding_cache[text] for text in self.document_texts
        ]
        for index, document in enumerate(self.documents):
            body = str(document.get("text", "")).strip()
            if body:
                self._embedding_cache.setdefault(body, self.document_embeddings[index])

    @staticmethod
    def document_text(document: Dict) -> str:
        """Combine the document title and body without domain-specific expansion."""
        title = str(document.get("title", "")).strip()
        text = str(document.get("text", "")).strip()
        return f"{title}\n{text}".strip()

    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is required for semantic retrieval. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc
        return SentenceTransformer(model_name)

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        encoded = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in encoded]
        if len(vectors) != len(texts):
            raise ValueError("The embedding model returned an unexpected vector count.")
        if any(not all(math.isfinite(value) for value in row) for row in vectors):
            raise ValueError("The embedding model returned a non-finite vector.")
        if vectors and len({len(vector) for vector in vectors}) != 1:
            raise ValueError("The embedding model returned inconsistent vector sizes.")
        return vectors

    @staticmethod
    def _cosine_from_normalized(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Cannot compare embedding vectors with different dimensions.")
        return sum(a * b for a, b in zip(left, right))

    def query(self, query_text: str, top_k: int = 6) -> List[Dict]:
        if not query_text.strip() or not self.documents or top_k <= 0:
            return []

        query_embedding = self._encode([query_text])[0]
        scores = [
            self._cosine_from_normalized(query_embedding, embedding)
            for embedding in self.document_embeddings
        ]
        ranked_indices = sorted(
            range(len(self.documents)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [
            {
                **self.documents[index],
                "raw_score": round(scores[index], 4),
            }
            for index in ranked_indices
        ]

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.similarities(text_a, [text_b])[0]

    def similarities(self, query_text: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        if not query_text.strip():
            return [0.0] * len(texts)

        usable_texts = [text if text.strip() else "" for text in texts]
        missing = list(dict.fromkeys(
            text for text in [query_text, *usable_texts]
            if text and text not in self._embedding_cache
        ))
        missing_embeddings = dict(zip(missing, self._encode(missing))) if missing else {}
        query_embedding = self._embedding_cache.get(
            query_text, missing_embeddings.get(query_text)
        )
        if query_embedding is None:
            raise ValueError("Could not create an embedding for the query.")

        scores = []
        for text in usable_texts:
            if not text:
                scores.append(0.0)
                continue
            embedding = self._embedding_cache.get(text, missing_embeddings.get(text))
            if embedding is None:
                raise ValueError("Could not create an embedding for a document.")
            score = self._cosine_from_normalized(query_embedding, embedding)
            scores.append(round(max(-1.0, min(1.0, score)), 4))
        return scores

    def pairwise_similarities(
        self, pairs: Sequence[tuple[str, str]]
    ) -> List[float]:
        if not pairs:
            return []
        texts = list(dict.fromkeys(
            text for pair in pairs for text in pair if text.strip()
        ))
        missing = [
            text for text in texts
            if text not in self._embedding_cache
        ]
        if missing:
            self._embedding_cache.update(zip(missing, self._encode(missing)))

        scores = []
        for left, right in pairs:
            if not left.strip() or not right.strip():
                scores.append(0.0)
                continue
            score = self._cosine_from_normalized(
                self._embedding_cache[left], self._embedding_cache[right]
            )
            scores.append(round(max(-1.0, min(1.0, score)), 4))
        return scores

    def top_terms_for_text(self, text: str, n: int = 4) -> List[str]:
        """Return frequent non-stopword terms for feedback-query expansion."""
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
            "was", "were", "what", "when", "where", "which", "who", "why",
            "with",
        }
        terms = [
            token for token in normalize_text(text).split()
            if len(token) > 2 and token not in stop_words
        ]
        counts = Counter(terms)
        return [term for term, _ in counts.most_common(max(0, n))]


# Keep the previous import name working for callers outside this repository.
TfidfIndex = SentenceTransformerIndex
