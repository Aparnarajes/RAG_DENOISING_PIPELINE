"""Domain-independent NLI contradiction detection with explainable source priority."""
from itertools import combinations
from typing import Dict, List, Tuple

from utils.evidence_analysis import (
    PairClassifier,
    SourceReliabilityPolicy,
    extract_claims,
    get_nli_comparator,
)

CONTRADICTION_CONFIDENCE_THRESHOLD = 0.65
CORROBORATION_CONFIDENCE_THRESHOLD = 0.65


class ContradictionDetectionAgent:
    def __init__(
        self,
        numeric_ratio_threshold: float | None = None,
        source_priorities: Dict[str, float] | None = None,
        nli_classifier: PairClassifier | None = None,
        contradiction_threshold: float = CONTRADICTION_CONFIDENCE_THRESHOLD,
        corroboration_threshold: float = CORROBORATION_CONFIDENCE_THRESHOLD,
    ):
        # Keep the former keyword temporarily accepted for older callers.
        del numeric_ratio_threshold
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("Contradiction threshold must be between 0 and 1.")
        if not 0.0 <= corroboration_threshold <= 1.0:
            raise ValueError("Corroboration threshold must be between 0 and 1.")
        self.source_policy = SourceReliabilityPolicy(source_priorities)
        self.nli = nli_classifier or get_nli_comparator()
        self.contradiction_threshold = contradiction_threshold
        self.corroboration_threshold = corroboration_threshold

    @staticmethod
    def _document_id(document: Dict) -> str:
        return str(document.get("id", document.get("document_id", "unknown")))

    def detect_and_resolve(
        self, documents: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        claims = [
            {
                "document_index": document_index,
                "document_id": self._document_id(document),
                "claim": claim,
            }
            for document_index, document in enumerate(documents)
            for claim in extract_claims(str(document.get("text", "")))
        ]
        candidate_pairs = [
            (left_index, right_index)
            for left_index, right_index in combinations(range(len(claims)), 2)
            if claims[left_index]["document_index"] != claims[right_index]["document_index"]
        ]

        directed_pairs = []
        for left_index, right_index in candidate_pairs:
            left, right = claims[left_index], claims[right_index]
            directed_pairs.extend([
                (left["claim"], right["claim"]),
                (right["claim"], left["claim"]),
            ])
        relations = self.nli.classify_pairs(directed_pairs) if directed_pairs else []
        directed_relations = {
            (left_index, right_index): relations[pair_index * 2]
            for pair_index, (left_index, right_index) in enumerate(candidate_pairs)
        }
        directed_relations.update({
            (right_index, left_index): relations[pair_index * 2 + 1]
            for pair_index, (left_index, right_index) in enumerate(candidate_pairs)
        })

        conflicts = []
        for left_index, right_index in candidate_pairs:
            left, right = claims[left_index], claims[right_index]
            left_to_right = directed_relations[(left_index, right_index)]
            right_to_left = directed_relations[(right_index, left_index)]
            contradiction = max(
                (
                    relation for relation in (left_to_right, right_to_left)
                    if relation["relationship"] == "CONTRADICTION"
                ),
                key=lambda relation: float(relation["confidence"]),
                default=None,
            )
            if (
                contradiction is None
                or float(contradiction["confidence"]) < self.contradiction_threshold
            ):
                continue

            left_document = documents[left["document_index"]]
            right_document = documents[right["document_index"]]
            left_priority = self.source_policy.score(left_document)
            right_priority = self.source_policy.score(right_document)
            left_corroborators = self._corroborators(
                left_index, claims, directed_relations, documents
            )
            right_corroborators = self._corroborators(
                right_index, claims, directed_relations, documents
            )

            preferred_index = None
            preferred_id = None
            if (
                left_priority > right_priority
                and any(priority >= right_priority for priority in left_corroborators.values())
            ):
                preferred_index = left["document_index"]
            elif (
                right_priority > left_priority
                and any(priority >= left_priority for priority in right_corroborators.values())
            ):
                preferred_index = right["document_index"]

            if preferred_index is not None:
                preferred_id = self._document_id(documents[preferred_index])
                corroborator_ids = (
                    left_corroborators if preferred_index == left["document_index"]
                    else right_corroborators
                )
                supporting_ids = [
                    document_id for document_id, priority in corroborator_ids.items()
                    if priority >= min(left_priority, right_priority)
                ]
                resolution = (
                    f"Preferred {preferred_id} (source priority "
                    f"{self.source_policy.score(documents[preferred_index]):.2f}) because "
                    f"independent trusted document(s) {', '.join(supporting_ids)} corroborate its claim."
                )
            else:
                resolution = (
                    "Conflict retained as unresolved: source priority alone or "
                    "uncorroborated evidence is insufficient to prefer either claim."
                )

            conflicts.append({
                "claim_a": left["claim"],
                "claim_b": right["claim"],
                "document_a": left["document_id"],
                "document_b": right["document_id"],
                "relationship": "CONTRADICTION",
                "confidence": round(float(contradiction["confidence"]), 3),
                "source_priority_a": round(left_priority, 3),
                "source_priority_b": round(right_priority, 3),
                "corroborating_documents_a": list(left_corroborators),
                "corroborating_documents_b": list(right_corroborators),
                "preferred_document": preferred_id,
                "resolution": resolution,
                # Preserve keys consumed by the existing dashboard and benchmark.
                "doc_a": left["document_id"],
                "doc_b": right["document_id"],
                "winner": preferred_id,
                "strategy": "source_priority_with_independent_corroboration" if preferred_id else "unresolved",
                "facet": left_document.get("facet", left_document.get("topic", "general")),
            })

        # Preserve every candidate; conflicts are reported, never silently dropped.
        resolved = list(documents)
        for document in resolved:
            document["stage"] = "contradiction_analyzed"
        return resolved, conflicts

    def _corroborators(
        self,
        claim_index: int,
        claims: List[Dict],
        directed_relations: Dict[Tuple[int, int], Dict],
        documents: List[Dict],
    ) -> Dict[str, float]:
        target = claims[claim_index]
        corroborators: Dict[str, float] = {}
        for other_index, other in enumerate(claims):
            if other_index == claim_index or other["document_index"] == target["document_index"]:
                continue
            document = documents[other["document_index"]]
            priority = self.source_policy.score(document)
            if priority < self.source_policy.min_trusted_score:
                continue
            relation = directed_relations.get((other_index, claim_index))
            if (
                relation
                and relation["relationship"] == "ENTAILMENT"
                and float(relation["confidence"]) >= self.corroboration_threshold
            ):
                document_id = self._document_id(document)
                corroborators[document_id] = max(
                    priority, corroborators.get(document_id, 0.0)
                )
        return corroborators
