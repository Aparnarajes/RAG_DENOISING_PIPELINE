"""Retrieval Agent: queries the sentence-transformer index and returns candidates."""
from typing import List, Dict
from utils.embeddings import SentenceTransformerIndex


class RetrievalAgent:
    """Fetches the top-k candidate documents for a query. Does no filtering —
    that responsibility belongs to downstream agents in the denoised pipeline."""

    def __init__(self, index: SentenceTransformerIndex):
        self.index = index

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        results = self.index.query(query, top_k=top_k)
        for r in results:
            r["stage"] = "retrieved"
        return results
