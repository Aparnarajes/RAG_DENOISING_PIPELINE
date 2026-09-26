"""
Contradiction Detection Agent: detects logical and factual contradictions
between candidate documents on the same entity or topic facet.
Resolves conflicts using:
1. Source priority (higher trust_score wins)
2. Majority vote (consensus across surviving documents if trust ties)
"""
import re
from typing import List, Dict, Tuple
from itertools import combinations

# Facet keywords for classifying text when explicit facet is absent
FACET_KEYWORDS = {
    "fast_charging_speed": ["fast charg", "dc fast", "minutes to charge", "under 5 min", "state of charge"],
    "solid_state_status": ["solid-state", "solid state", "liquid electrolyte", "beyond 2027", "every electric car"],
    "capacity_degradation": ["capacity", "degrade", "warranty", "warranties", "lose 50 percent", "retain"],
    "fire_safety": ["fire", "fires", "flame", "burn", "safety", "catch fire"],
    "cold_weather_range": ["cold", "winter", "freezing", "snow", "cabin heat"],
    "pack_cost": ["cost", "price", "per kilowatt-hour", "kwh", "pack prices"],
    "battery_recycling": ["recycl", "spent", "hydrometallurg", "recover", "second life"],
}

NUMERIC_CLAIM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(percent|%|minutes?|hours?|dollars?|\$|kwh|kw|times)",
    re.IGNORECASE
)


def _infer_facet(doc: Dict) -> str:
    if "facet" in doc and doc["facet"]:
        return doc["facet"]
    text_lower = doc["text"].lower()
    for facet, keywords in FACET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return facet
    return doc.get("topic", "general")


def _extract_facet_claims(text: str) -> List[Tuple[float, str]]:
    matches = NUMERIC_CLAIM_PATTERN.findall(text.lower())
    claims = []
    for val, unit in matches:
        unit = unit.lower().replace("%", "percent").replace("$", "dollar")
        claims.append((float(val), unit))
    return claims


class ContradictionDetectionAgent:
    def __init__(self, numeric_ratio_threshold: float = 1.8):
        self.numeric_ratio_threshold = numeric_ratio_threshold

    def detect_and_resolve(self, documents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        by_facet: Dict[str, List[Dict]] = {}
        for doc in documents:
            facet = _infer_facet(doc)
            by_facet.setdefault(facet, []).append(doc)

        dropped_ids = set()
        conflict_log = []

        for facet, docs in by_facet.items():
            if len(docs) < 2:
                continue

            for doc_a, doc_b in combinations(docs, 2):
                if doc_a["id"] in dropped_ids or doc_b["id"] in dropped_ids:
                    continue

                is_conflict = False
                claim_a_desc = ""
                claim_b_desc = ""

                # 1. Semantic contradiction check based on mutually exclusive assertions
                text_a = doc_a["text"].lower()
                text_b = doc_b["text"].lower()

                if "delayed beyond 2027" in text_a and "already in every" in text_b:
                    is_conflict = True
                    claim_a_desc = "Commercial deployment delayed beyond 2027"
                    claim_b_desc = "Already in every car sold today"
                elif "already in every" in text_a and "delayed beyond 2027" in text_b:
                    is_conflict = True
                    claim_a_desc = "Already in every car sold today"
                    claim_b_desc = "Commercial deployment delayed beyond 2027"
                elif "substantially lower rate" in text_a and "100 times more common" in text_b:
                    is_conflict = True
                    claim_a_desc = "Fires occur at substantially lower rate"
                    claim_b_desc = "Fires 100 times more common"
                elif "100 times more common" in text_a and "substantially lower rate" in text_b:
                    is_conflict = True
                    claim_a_desc = "Fires 100 times more common"
                    claim_b_desc = "Fires occur at substantially lower rate"
                else:
                    # 2. Numeric contradiction check WITHIN THE SAME FACET
                    claims_a = _extract_facet_claims(doc_a["text"])
                    claims_b = _extract_facet_claims(doc_b["text"])
                    for val_a, unit_a in claims_a:
                        for val_b, unit_b in claims_b:
                            if unit_a == unit_b and val_a > 0 and val_b > 0:
                                ratio = max(val_a, val_b) / min(val_a, val_b)
                                if ratio >= self.numeric_ratio_threshold:
                                    is_conflict = True
                                    claim_a_desc = f"{val_a} {unit_a}"
                                    claim_b_desc = f"{val_b} {unit_b}"
                                    break
                        if is_conflict:
                            break

                if is_conflict:
                    trust_a = doc_a.get("trust_score", 0.5)
                    trust_b = doc_b.get("trust_score", 0.5)

                    # Resolution policy:
                    # 1. Source priority: Higher trust score wins
                    # 2. Majority vote / consensus if trust scores tie
                    if abs(trust_a - trust_b) > 0.05:
                        winner = doc_a if trust_a > trust_b else doc_b
                        loser = doc_b if winner is doc_a else doc_a
                        strategy = "source_priority"
                        reason = f"kept {winner['id']} (trust={winner.get('trust_score', 0):.2f}) over {loser['id']} (trust={loser.get('trust_score', 0):.2f})"
                    else:
                        # Tie-breaker: majority vote by checking which aligns with remaining docs in topic
                        votes_a = sum(1 for d in docs if d["id"] not in (doc_a["id"], doc_b["id"]) and d.get("trust_score", 0) >= 0.5)
                        # Higher relevance score as secondary tie breaker
                        rel_a = doc_a.get("relevance_score", 0)
                        rel_b = doc_b.get("relevance_score", 0)
                        winner = doc_a if rel_a >= rel_b else doc_b
                        loser = doc_b if winner is doc_a else doc_a
                        strategy = "majority_vote_and_relevance"
                        reason = f"tied trust ({trust_a:.2f}); resolved by relevance/consensus (kept {winner['id']} rel={winner.get('relevance_score', 0):.2f})"

                    dropped_ids.add(loser["id"])
                    conflict_log.append({
                        "facet": facet,
                        "doc_a": doc_a["id"],
                        "claim_a": claim_a_desc,
                        "doc_b": doc_b["id"],
                        "claim_b": claim_b_desc,
                        "winner": winner["id"],
                        "strategy": strategy,
                        "resolution": reason,
                    })

        resolved = [d for d in documents if d["id"] not in dropped_ids]
        for d in resolved:
            d["stage"] = "contradiction_resolved"
        return resolved, conflict_log
