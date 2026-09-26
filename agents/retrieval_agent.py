"""Retrieval Agent: queries the vector/TF-IDF index and returns raw candidates."""
from typing import List, Dict
from utils.embeddings import TfidfIndex


class RetrievalAgent:
    """Fetches the top-k candidate documents for a query. Does no filtering —
    that responsibility belongs to downstream agents in the denoised pipeline."""

    def __init__(self, index: TfidfIndex):
        self.index = index

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        results = self.index.query(query, top_k=top_k)
        for r in results:
            r["stage"] = "retrieved"
        return results
