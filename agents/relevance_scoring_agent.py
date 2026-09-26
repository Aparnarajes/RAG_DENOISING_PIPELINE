"""
Relevance Scoring Agent: assigns a 0-1 relevance score to each retrieved
document using semantic similarity plus query-alignment heuristics (keyword
overlap and query-facet intent alignment), then filters out anything below a
configurable threshold.
"""
from typing import List, Dict
from utils.embeddings import TfidfIndex, normalize_text

INTENT_MAP = {
    "cost": ["cost", "costs", "price", "prices", "dollar", "dollars", "kwh", "kilowatt"],
    "charging": ["charge", "charging", "charger", "fast", "minutes", "minute"],
    "solid_state": ["solid-state", "solid state", "electrolyte"],
    "degradation": ["capacity", "degrade", "degradation", "retention", "retain", "warranty", "warranties"],
    "fire": ["fire", "fires", "catch fire", "burn"],
    "recycling": ["retired", "retir", "retire", "spent", "recycl", "recycling", "end-of-life", "recover", "nickel", "lithium"],
    "winter": ["weather", "cold", "winter", "freezing", "temperature", "cabin heat"],
    "climate": ["climate", "warming", "greenhouse", "carbon", "fossil", "emissions", "atmosphere"],
}


class RelevanceScoringAgent:
    def __init__(self, index: TfidfIndex, threshold: float = 0.20):
        self.index = index
        self.threshold = threshold

    def _keyword_overlap(self, query: str, text: str) -> float:
        q_norm = normalize_text(query, expand_synonyms=True).split()
        d_norm = set(normalize_text(text).split())
        q_words = [w for w in q_norm if len(w) > 3]
        if not q_words:
            return 0.0
        hits = sum(1 for w in q_words if w in d_norm)
        return min(1.0, hits / len(set(q_words)))

    def _intent_alignment(self, query: str, text: str) -> tuple[float, str]:
        q_norm = normalize_text(query).lower()
        t_norm = normalize_text(text).lower()

        # Identify which intent(s) the query expresses
        matched_intents = [
            intent for intent, words in INTENT_MAP.items()
            if any(w in q_norm for w in words)
        ]

        if not matched_intents:
            return 0.0, "general query intent"

        # Check if document satisfies the query intent
        doc_matches = [
            intent for intent in matched_intents
            if any(w in t_norm for w in INTENT_MAP[intent])
        ]

        if doc_matches:
            return 0.12, f"matches query intent ({', '.join(doc_matches)})"
        else:
            return -0.25, f"misses core query intent ({', '.join(matched_intents)})"

    def score(self, query: str, documents: List[Dict]) -> List[Dict]:
        scored = []
        for doc in documents:
            sim = self.index.similarity(query, doc["text"])
            overlap = self._keyword_overlap(query, doc["text"])
            intent_delta, intent_desc = self._intent_alignment(query, doc["text"])

            # Blended relevance score (50% semantic similarity + 40% keyword overlap + intent alignment)
            base_score = 0.50 * sim + 0.40 * overlap
            final_score = round(base_score + intent_delta, 3)
            final_score = max(0.0, min(1.0, final_score))

            reasoning = (
                f"sim={sim:.3f}, overlap={overlap:.3f}, intent={intent_desc} "
                f"-> relevance={final_score:.3f} "
                f"({'PASS' if final_score >= self.threshold else 'FILTERED' } threshold {self.threshold})"
            )
            scored.append({
                **doc,
                "relevance_score": final_score,
                "relevance_reasoning": reasoning,
                "stage": "scored",
            })
        return sorted(scored, key=lambda d: d["relevance_score"], reverse=True)

    def filter(self, scored_documents: List[Dict]) -> List[Dict]:
        kept = [d for d in scored_documents if d["relevance_score"] >= self.threshold]
        for d in kept:
            d["stage"] = "relevance_filtered"
        return kept
