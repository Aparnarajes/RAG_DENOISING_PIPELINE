"""Claim-level grounding against trusted, verified evidence."""
from typing import Dict, List

from utils.evidence_analysis import (
    PairClassifier,
    SourceReliabilityPolicy,
    extract_claims,
    get_nli_comparator,
)

CLAIM_CONFIDENCE_THRESHOLD = 0.65


class HallucinationDetectionAgent:
    def __init__(
        self,
        nli_classifier: PairClassifier | None = None,
        source_priorities: Dict[str, float] | None = None,
        entailment_threshold: float = CLAIM_CONFIDENCE_THRESHOLD,
    ):
        if not 0.0 <= entailment_threshold <= 1.0:
            raise ValueError("Entailment threshold must be between 0 and 1.")
        self.nli = nli_classifier or get_nli_comparator()
        self.source_policy = SourceReliabilityPolicy(source_priorities)
        self.entailment_threshold = entailment_threshold

    @staticmethod
    def _verified_evidence(
        documents: List[Dict], source_policy: SourceReliabilityPolicy
    ) -> List[Dict]:
        evidence = []
        for document in documents:
            priority = source_policy.score(document)
            if priority < source_policy.min_trusted_score:
                continue

            document_id = document.get("id", document.get("document_id"))
            verification_claims = document.get("verification_claims")
            if verification_claims is not None:
                evidence.extend(
                    {
                        "document_id": document_id,
                        "claim": claim["claim"],
                        "source_priority": priority,
                    }
                    for claim in verification_claims
                    if claim.get("status") == "VERIFIED"
                )
            elif document.get("verification_status") not in {
                "UNSUPPORTED",
                "CONTRADICTED",
            }:
                evidence.extend(
                    {
                        "document_id": document_id,
                        "claim": claim,
                        "source_priority": priority,
                    }
                    for claim in extract_claims(str(document.get("text", "")))
                )
        return evidence

    @staticmethod
    def _probabilities(relation: Dict) -> Dict[str, float]:
        raw = relation.get("probabilities")
        if isinstance(raw, dict):
            return {
                str(label).upper(): max(0.0, min(1.0, float(probability)))
                for label, probability in raw.items()
            }
        relationship = str(relation.get("relationship", "NEUTRAL")).upper()
        confidence = max(0.0, min(1.0, float(relation.get("confidence", 0.0))))
        return {relationship: confidence}

    def check(
        self,
        answer_text: str,
        used_documents: List[Dict],
        verified_corpus: List[Dict] | None = None,
    ) -> Dict:
        answer_claims = extract_claims(answer_text)
        documents = list(used_documents)
        if verified_corpus:
            documents.extend(verified_corpus)
        evidence = self._verified_evidence(documents, self.source_policy)
        pairs = [
            (item["claim"], answer_claim)
            for answer_claim in answer_claims
            for item in evidence
        ]
        relationships = self.nli.classify_pairs(pairs) if pairs else []
        if len(relationships) != len(pairs):
            raise ValueError("NLI classifier returned an unexpected number of results.")

        claim_results = []
        evidence_count = len(evidence)
        for claim_index, answer_claim in enumerate(answer_claims):
            matches = []
            for evidence_index, evidence_item in enumerate(evidence):
                relation = relationships[claim_index * evidence_count + evidence_index]
                relation_name = str(relation["relationship"]).upper()
                relation_confidence = max(
                    0.0, min(1.0, float(relation.get("confidence", 0.0)))
                )
                matches.append({
                    **evidence_item,
                    "relationship": relation_name,
                    "nli_confidence": relation_confidence,
                    "nli_probabilities": self._probabilities(relation),
                })

            supports = [
                match for match in matches
                if match["relationship"] == "ENTAILMENT"
                and match["nli_confidence"] >= self.entailment_threshold
            ]
            contradictions = [
                match for match in matches
                if match["relationship"] == "CONTRADICTION"
                and match["nli_confidence"] >= self.entailment_threshold
            ]
            best_support = max(
                supports,
                key=lambda match: (
                    match["source_priority"],
                    match["nli_confidence"],
                ),
                default=None,
            )
            best_contradiction = max(
                contradictions,
                key=lambda match: (
                    match["source_priority"],
                    match["nli_confidence"],
                ),
                default=None,
            )

            if best_contradiction and (
                best_support is None
                or best_contradiction["source_priority"]
                >= best_support["source_priority"]
            ):
                status = "CONTRADICTED"
                if best_support is None:
                    selected_matches = contradictions
                else:
                    selected_matches = [
                        match for match in contradictions
                        if match["source_priority"] >= best_support["source_priority"]
                    ]
            elif supports:
                status = "SUPPORTED"
                selected_matches = supports
            else:
                status = "UNSUPPORTED"
                selected_matches = []

            selected = max(
                selected_matches,
                key=lambda match: match["nli_confidence"],
                default=None,
            )
            if selected is not None:
                classification_confidence = selected["nli_confidence"]
                probabilities = selected["nli_probabilities"]
            elif matches:
                selected = max(matches, key=lambda match: match["nli_confidence"])
                classification_confidence = selected["nli_confidence"]
                probabilities = selected["nli_probabilities"]
            else:
                classification_confidence = 0.0
                probabilities = {}

            evidence_ids = list(dict.fromkeys(
                str(match["document_id"])
                for match in selected_matches
                if match["document_id"] is not None
            ))
            evidence_records = [
                {
                    "document_id": match["document_id"],
                    "claim": match["claim"],
                    "relationship": match["relationship"],
                    "nli_confidence": match["nli_confidence"],
                    "nli_probabilities": match["nli_probabilities"],
                }
                for match in selected_matches
            ]
            claim_results.append({
                "claim": answer_claim,
                "status": status,
                "confidence": round(classification_confidence, 3),
                "nli_confidence": round(classification_confidence, 3),
                "nli_probabilities": probabilities,
                "evidence_document_ids": evidence_ids,
                "evidence": evidence_records,
                # Compatibility with existing consumers of the detector result.
                "source_document_ids": evidence_ids,
            })

        checked = len(claim_results)
        supported = sum(claim["status"] == "SUPPORTED" for claim in claim_results)
        flags = [
            {
                "sentence": claim["claim"],
                "issue_type": (
                    "CONTRADICTED_BY_EVIDENCE"
                    if claim["status"] == "CONTRADICTED"
                    else "UNSUPPORTED_BY_EVIDENCE"
                ),
                "reason": (
                    f"Evidence from {', '.join(claim['evidence_document_ids'])} "
                    "supports an opposing claim."
                    if claim["status"] == "CONTRADICTED"
                    else "No sufficiently confident verified evidence supports this claim."
                ),
                "confidence": claim["confidence"],
            }
            for claim in claim_results
            if claim["status"] != "SUPPORTED"
        ]
        supported_ratio = round(supported / checked, 3) if checked else 1.0
        return {
            "claims": claim_results,
            "flags": flags,
            "supported_ratio": supported_ratio,
            "hallucination_rate": round(1.0 - supported_ratio, 3),
            "sentences_checked": checked,
            "supported_count": supported,
            "flagged_count": len(flags),
        }
