"""
Generic relevance scoring using semantic similarity and query-term alignment.
"""
import math
import os
from typing import Dict, List

from utils.embeddings import SentenceTransformerIndex, normalize_text

DEFAULT_RELEVANCE_THRESHOLD = 0.45
SEMANTIC_WEIGHT = 0.8
ALIGNMENT_WEIGHT = 0.2
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


class RelevanceScoringAgent:
    def __init__(
        self,
        index: SentenceTransformerIndex,
        threshold: float | None = None,
    ):
        if threshold is None:
            configured_threshold = os.getenv("RAG_RELEVANCE_THRESHOLD")
            threshold = (
                float(configured_threshold)
                if configured_threshold is not None
                else DEFAULT_RELEVANCE_THRESHOLD
            )
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Relevance threshold must be a finite number from 0 to 1.")
        self.index = index
        self.threshold = threshold

    @staticmethod
    def _query_alignment(query: str, document_text: str) -> float:
        query_terms = {
            term for term in normalize_text(query).split()
            if len(term) > 2 and term not in STOP_WORDS
        }
        if not query_terms:
            return 0.0
        document_terms = set(normalize_text(document_text).split())
        return len(query_terms & document_terms) / len(query_terms)

    def score(self, query: str, documents: List[Dict]) -> List[Dict]:
        documents_needing_similarity = [
            index for index, document in enumerate(documents)
            if document.get("raw_score") is None
        ]
        similarities = iter(self.index.similarities(
            query,
            [
                self.index.document_text(documents[index])
                for index in documents_needing_similarity
            ],
        ))
        similarity_by_index = {
            index: next(similarities) for index in documents_needing_similarity
        }

        scored = []
        for index, document in enumerate(documents):
            document_text = self.index.document_text(document)
            semantic_similarity = document.get("raw_score")
            if semantic_similarity is None:
                semantic_similarity = similarity_by_index[index]
            semantic_similarity = max(0.0, min(1.0, float(semantic_similarity)))
            query_alignment = self._query_alignment(query, document_text)
            score = round(
                SEMANTIC_WEIGHT * semantic_similarity
                + ALIGNMENT_WEIGHT * query_alignment,
                3,
            )
            decision = "KEEP" if score >= self.threshold else "REMOVE"
            reasoning = (
                f"semantic similarity={semantic_similarity:.3f} "
                f"({SEMANTIC_WEIGHT:.0%}), query-term alignment={query_alignment:.3f} "
                f"({ALIGNMENT_WEIGHT:.0%}); score={score:.3f}, "
                f"threshold={self.threshold:.3f} -> {decision}"
            )
            document_id = document.get("id", document.get("document_id"))
            scored.append({
                **document,
                "document_id": document_id,
                "score": score,
                "reasoning": reasoning,
                "decision": decision,
                # Retain the established names consumed by pipeline and benchmark.
                "relevance_score": score,
                "relevance_reasoning": reasoning,
                "stage": "scored",
            })
        return sorted(scored, key=lambda document: document["score"], reverse=True)

    def filter(self, scored_documents: List[Dict]) -> List[Dict]:
        kept = [
            document for document in scored_documents
            if document.get(
                "decision",
                "KEEP" if document.get("relevance_score", 0.0) >= self.threshold else "REMOVE",
            ) == "KEEP"
        ]
        for document in kept:
            document["stage"] = "relevance_filtered"
        return kept
