"""Evidence-grounded claim verification with explicit source-quality metadata."""
from typing import Dict, List

from utils.evidence_analysis import (
    MIN_VERIFIED_SOURCE_SCORE,
    PairClassifier,
    SourceReliabilityPolicy,
    extract_claims,
    get_nli_comparator,
)

VERIFICATION_CONFIDENCE_THRESHOLD = 0.65


class EvidenceVerificationAgent:
    def __init__(
        self,
        trust_threshold: float = MIN_VERIFIED_SOURCE_SCORE,
        source_priorities: Dict[str, float] | None = None,
        nli_classifier: PairClassifier | None = None,
        entailment_threshold: float = VERIFICATION_CONFIDENCE_THRESHOLD,
    ):
        if not 0.0 <= trust_threshold <= 1.0:
            raise ValueError("Trust threshold must be between 0 and 1.")
        if not 0.0 <= entailment_threshold <= 1.0:
            raise ValueError("Entailment threshold must be between 0 and 1.")
        self.trust_threshold = trust_threshold
        self.source_policy = SourceReliabilityPolicy(
            source_priorities, min_trusted_score=trust_threshold
        )
        self.nli = nli_classifier or get_nli_comparator()
        self.entailment_threshold = entailment_threshold

    def verify(self, documents: List[Dict]) -> List[Dict]:
        document_claims = [
            (document_index, claim)
            for document_index, document in enumerate(documents)
            for claim in extract_claims(str(document.get("text", "")))
        ]
        trusted_evidence = [
            (document_index, document.get("id", document.get("document_id")), claim)
            for document_index, document in enumerate(documents)
            if self.source_policy.score(document) >= self.trust_threshold
            for claim in extract_claims(str(document.get("text", "")))
        ]

        pair_records = [
            (claim_index, evidence_index, (evidence_claim, claim))
            for claim_index, (_, claim) in enumerate(document_claims)
            for evidence_index, (source_index, _, evidence_claim) in enumerate(trusted_evidence)
        ]
        relationships = self.nli.classify_pairs(
            [pair for _, _, pair in pair_records]
        ) if pair_records else []
        if len(relationships) != len(pair_records):
            raise ValueError("The NLI classifier returned an unexpected result count.")
        relation_by_claim: Dict[int, List[Dict]] = {
            index: [] for index in range(len(document_claims))
        }
        for (claim_index, evidence_index, _), relation in zip(pair_records, relationships):
            source_index, source_id, evidence_claim = trusted_evidence[evidence_index]
            relation_by_claim[claim_index].append({
                "source_index": source_index,
                "source_id": source_id,
                "evidence_claim": evidence_claim,
                "relationship": relation["relationship"],
                "confidence": float(relation["confidence"]),
            })

        claims_by_document: Dict[int, List[Dict]] = {
            index: [] for index in range(len(documents))
        }
        for claim_index, (document_index, claim) in enumerate(document_claims):
            matches = relation_by_claim[claim_index]
            supports = [
                match for match in matches
                if match["relationship"] == "ENTAILMENT"
                and match["confidence"] >= self.entailment_threshold
            ]
            contradictions = [
                match for match in matches
                if match["relationship"] == "CONTRADICTION"
                and match["confidence"] >= self.entailment_threshold
            ]
            best_support = max(
                supports,
                key=lambda match: (
                    self.source_policy.score(documents[match["source_index"]]),
                    match["confidence"],
                ),
                default=None,
            )
            best_contradiction = max(
                contradictions,
                key=lambda match: (
                    self.source_policy.score(documents[match["source_index"]]),
                    match["confidence"],
                ),
                default=None,
            )

            if best_contradiction and (
                best_support is None
                or self.source_policy.score(documents[best_contradiction["source_index"]])
                >= self.source_policy.score(documents[best_support["source_index"]])
            ):
                status = "CONTRADICTED"
                chosen_evidence = best_contradiction
            elif best_support:
                status = "VERIFIED"
                chosen_evidence = best_support
            else:
                status = "UNSUPPORTED"
                chosen_evidence = None

            claims_by_document[document_index].append({
                "claim": claim,
                "source_document_ids": (
                    [chosen_evidence["source_id"]] if chosen_evidence else []
                ),
                "status": status,
                "evidence": (
                    chosen_evidence["evidence_claim"] if chosen_evidence else ""
                ),
                "confidence": (
                    round(chosen_evidence["confidence"], 3) if chosen_evidence else 0.0
                ),
            })

        verified = []
        for index, document in enumerate(documents):
            claims = claims_by_document[index]
            source_score = self.source_policy.score(document)
            if not claims or all(claim["status"] == "UNSUPPORTED" for claim in claims):
                verification_status = "UNSUPPORTED"
            elif any(claim["status"] == "CONTRADICTED" for claim in claims):
                verification_status = "CONTRADICTED"
            elif all(claim["status"] == "VERIFIED" for claim in claims):
                verification_status = "VERIFIED"
            else:
                verification_status = "UNSUPPORTED"

            if source_score < self.trust_threshold:
                reasoning = (
                    f"Source score {source_score:.2f} is below trusted threshold "
                    f"{self.trust_threshold:.2f}; document claims cannot serve as verified evidence."
                )
            else:
                reasoning = (
                    f"{sum(claim['status'] == 'VERIFIED' for claim in claims)} of "
                    f"{len(claims)} extracted claims supported by trusted evidence; "
                    f"document verification status={verification_status}."
                )

            verified.append({
                **document,
                "trust_score": round(source_score, 3),
                "source_category": document.get(
                    "source_category",
                    document.get("source_type", "legacy_trusted" if source_score else "unknown"),
                ),
                "verification_status": verification_status,
                "verification_claims": claims,
                "verification_reasoning": reasoning,
                "stage": "verified",
            })
        return verified

    def filter(self, verified_documents: List[Dict]) -> List[Dict]:
        kept = [
            document for document in verified_documents
            if (
                document["trust_score"] >= self.trust_threshold
                and document["verification_status"] == "VERIFIED"
            )
        ]
        for document in kept:
            document["stage"] = "evidence_filtered"
        return kept
