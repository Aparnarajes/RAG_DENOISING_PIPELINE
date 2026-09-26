import re
import unittest
from types import SimpleNamespace

from agents.contradiction_detection_agent import ContradictionDetectionAgent
from agents.evidence_verification_agent import EvidenceVerificationAgent
from agents.hallucination_detection_agent import HallucinationDetectionAgent
from utils.evidence_analysis import NLIComparator, SourceReliabilityPolicy


class FakeNLIClassifier:
    """Deterministic NLI test double; production behavior uses a local NLI model."""

    RELATIONAL_OPPOSITES = {
        "increase": "decrease",
        "increases": "decreases",
        "increased": "decreased",
        "increasing": "decreasing",
        "decrease": "increase",
        "decreases": "increases",
        "decreased": "increased",
        "decreasing": "increasing",
        "rise": "fall",
        "rises": "falls",
        "fall": "rise",
        "falls": "rises",
        "improve": "worsen",
        "improves": "worsens",
        "improved": "worsened",
        "worsen": "improve",
        "worsens": "improves",
        "worsened": "improved",
        "support": "oppose",
        "supports": "opposes",
        "oppose": "support",
        "opposes": "supports",
        "enable": "prevent",
        "enables": "prevents",
        "prevent": "enable",
        "prevents": "enables",
    }
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "be", "by", "does", "for", "from",
        "in", "is", "of", "on", "the", "to", "what", "when", "with",
    }

    @staticmethod
    def _tokens(text):
        return set(re.findall(r"\b\w+\b", text.lower()))

    def classify_pairs(self, pairs):
        results = []
        for premise, hypothesis in pairs:
            premise_tokens = self._tokens(premise)
            hypothesis_tokens = self._tokens(hypothesis)
            shared_content = (
                premise_tokens & hypothesis_tokens
            ) - self.STOP_WORDS
            has_opposites = any(
                self.RELATIONAL_OPPOSITES.get(token) in hypothesis_tokens
                for token in premise_tokens
                if token in self.RELATIONAL_OPPOSITES
            ) or any(
                self.RELATIONAL_OPPOSITES.get(token) in premise_tokens
                for token in hypothesis_tokens
                if token in self.RELATIONAL_OPPOSITES
            )
            if has_opposites and shared_content:
                relationship, confidence = "CONTRADICTION", 0.96
            elif premise_tokens == hypothesis_tokens:
                relationship, confidence = "ENTAILMENT", 0.99
            elif shared_content and len(shared_content) >= 2:
                relationship, confidence = "ENTAILMENT", 0.9
            else:
                relationship, confidence = "NEUTRAL", 0.9
            results.append({
                "relationship": relationship,
                "confidence": confidence,
            })
        return results


class FakeCrossEncoder:
    def __init__(self, scores=None):
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            )
        )
        self.scores = scores

    def predict(self, pairs, **kwargs):
        return self.scores or [[0.99, 0.005, 0.005] for _ in pairs]


class FakeSemanticIndex:
    def pairwise_similarities(self, pairs):
        return [
            0.05 if "saturn" in right.lower() else 0.8
            for _, right in pairs
        ]


def trusted_document(document_id, text, source_category="academic_research"):
    return {
        "id": document_id,
        "text": text,
        "source": document_id,
        "source_category": source_category,
    }


class GenericEvidenceAgentTests(unittest.TestCase):
    def setUp(self):
        self.nli = FakeNLIClassifier()

    def verify_with_independent_support(self, claim):
        return EvidenceVerificationAgent(nli_classifier=self.nli).verify([
            trusted_document("SOURCE1", claim),
            trusted_document("SOURCE2", claim),
        ])[0]

    def test_nli_semantic_guard_rejects_unrelated_pair_as_contradiction(self):
        comparator = NLIComparator(
            model=FakeCrossEncoder(), semantic_index=FakeSemanticIndex()
        )

        unrelated, related = comparator.classify_pairs([
            (
                "Photosynthesis lets plants convert sunlight into chemical energy.",
                "Saturn contains a hidden city beneath its northern clouds.",
            ),
            ("X increases Y.", "X decreases Y."),
        ])

        self.assertEqual(unrelated["relationship"], "NEUTRAL")
        self.assertEqual(related["relationship"], "CONTRADICTION")

    def test_nli_output_maps_entailment_contradiction_and_neutral(self):
        comparator = NLIComparator(
            model=FakeCrossEncoder([
                [0.01, 0.98, 0.01],
                [0.98, 0.01, 0.01],
                [0.01, 0.01, 0.98],
            ]),
            semantic_index=FakeSemanticIndex(),
        )

        results = comparator.classify_pairs([
            ("An adult human has two lungs.", "A person has two lungs."),
            ("X increases Y.", "X decreases Y."),
            (
                "Photosynthesis lets plants convert sunlight into chemical energy.",
                "Saturn contains a hidden city beneath its northern clouds.",
            ),
        ])

        self.assertEqual(
            [result["relationship"] for result in results],
            ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"],
        )
        self.assertAlmostEqual(results[0]["confidence"], 0.98)
        self.assertIn("ENTAILMENT", results[0]["probabilities"])
        self.assertLess(results[2]["semantic_similarity"], 0.2)

    def test_verification_supports_ai_claim(self):
        claim = "Supervised learning trains models with labeled examples."
        result = self.verify_with_independent_support(claim)

        self.assertEqual(result["verification_claims"][0]["status"], "VERIFIED")
        self.assertEqual(result["verification_claims"][0]["source_document_ids"], ["SOURCE1"])

    def test_verification_supports_science_claim(self):
        claim = "Photosynthesis lets plants use sunlight to make sugars."
        result = self.verify_with_independent_support(claim)

        self.assertEqual(result["verification_claims"][0]["status"], "VERIFIED")

    def test_verification_supports_health_claim(self):
        claim = "Caffeine can affect sleep by increasing alertness."
        result = self.verify_with_independent_support(claim)

        self.assertEqual(result["verification_claims"][0]["status"], "VERIFIED")

    def test_verification_supports_ev_claim_without_ev_rules(self):
        claim = "Cold weather can reduce electric vehicle battery range."
        result = self.verify_with_independent_support(claim)

        self.assertEqual(result["verification_claims"][0]["status"], "VERIFIED")

    def test_unsupported_claim_is_not_marked_verified(self):
        verifier = EvidenceVerificationAgent(nli_classifier=self.nli)
        result = self.verify_with_independent_support(
            "Photosynthesis lets plants use sunlight to make sugars."
        )

        self.assertEqual(result["verification_claims"][0]["status"], "VERIFIED")
        unknown_claim = "Saturn contains a hidden city beneath its northern clouds."
        unsupported_document = verifier.verify([
            trusted_document("IRRELEVANT", "Photosynthesis lets plants use sunlight to make sugars."),
            trusted_document("UNKNOWN", unknown_claim, "user_generated_forum"),
        ])[1]
        self.assertEqual(
            unsupported_document["verification_claims"][0]["status"], "UNSUPPORTED"
        )
        self.assertEqual(verifier.filter([unsupported_document]), [])

        unsupported = HallucinationDetectionAgent(nli_classifier=self.nli).check(
            unknown_claim,
            [trusted_document("IRRELEVANT", "Photosynthesis lets plants use sunlight to make sugars.")],
        )
        self.assertEqual(unsupported["claims"][0]["status"], "UNSUPPORTED")
        self.assertEqual(unsupported["flags"][0]["issue_type"], "UNSUPPORTED_BY_EVIDENCE")

    def test_untrusted_claim_cannot_verify_itself(self):
        claim = "Caffeine can affect sleep by increasing alertness."
        result = EvidenceVerificationAgent(nli_classifier=self.nli).verify([
            trusted_document("FORUM1", claim, "user_generated_forum")
        ])[0]

        self.assertEqual(result["trust_score"], 0.2)
        self.assertEqual(result["verification_claims"][0]["status"], "UNSUPPORTED")
        self.assertEqual(EvidenceVerificationAgent(nli_classifier=self.nli).filter([result]), [])

    def test_low_priority_unsupported_claim_is_rejected_from_evidence(self):
        verifier = EvidenceVerificationAgent(nli_classifier=self.nli)
        unsupported = verifier.verify([
            trusted_document("SCI1", "Photosynthesis lets plants use sunlight to make sugars."),
            {
                **trusted_document(
                    "OTHER",
                    "Saturn contains a hidden city beneath its clouds.",
                    "user_generated_forum",
                ),
            }
        ])[1:]

        self.assertEqual(unsupported[0]["verification_status"], "UNSUPPORTED")
        self.assertEqual(verifier.filter(unsupported), [])

    def test_claim_record_has_required_structure(self):
        claim = "Photosynthesis lets plants use sunlight to make sugars."
        record = EvidenceVerificationAgent(nli_classifier=self.nli).verify([
            trusted_document("SCI1", claim),
            trusted_document("SCI2", claim),
        ])[0]["verification_claims"][0]

        self.assertEqual(
            set(record),
            {"claim", "source_document_ids", "status", "evidence", "confidence"},
        )
        self.assertGreaterEqual(record["confidence"], 0.0)
        self.assertLessEqual(record["confidence"], 1.0)

    def test_contradictions_are_detected_and_both_documents_retained(self):
        documents = [
            trusted_document("D1", "X increases Y."),
            trusted_document("D2", "X decreases Y.", "user_generated_forum"),
        ]

        retained, conflicts = ContradictionDetectionAgent(
            nli_classifier=self.nli
        ).detect_and_resolve(documents)

        self.assertEqual([doc["id"] for doc in retained], ["D1", "D2"])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["relationship"], "CONTRADICTION")
        self.assertEqual(conflicts[0]["document_a"], "D1")
        self.assertEqual(conflicts[0]["document_b"], "D2")
        self.assertEqual(conflicts[0]["claim_a"], "X increases Y.")
        self.assertEqual(conflicts[0]["claim_b"], "X decreases Y.")

    def test_source_priority_requires_independent_corroboration_and_explains_choice(self):
        documents = [
            trusted_document("OFFICIAL", "X increases Y.", "official/government"),
            trusted_document("FORUM", "X decreases Y.", "user-generated/forum"),
            trusted_document("RESEARCH", "X increases Y.", "academic/research"),
        ]

        _, conflicts = ContradictionDetectionAgent(
            nli_classifier=self.nli
        ).detect_and_resolve(documents)
        target_conflict = next(
            conflict for conflict in conflicts
            if {conflict["document_a"], conflict["document_b"]} == {"OFFICIAL", "FORUM"}
        )

        self.assertEqual(target_conflict["preferred_document"], "OFFICIAL")
        self.assertIn("RESEARCH", target_conflict["resolution"])
        self.assertIn("independent trusted", target_conflict["resolution"])

    def test_source_priority_categories_are_configurable(self):
        policy = SourceReliabilityPolicy({
            "official/government": 0.6,
            "user-generated/forum": 0.95,
        })

        self.assertEqual(
            policy.score(trusted_document("OFFICIAL", "Claim.", "official/government")),
            0.6,
        )
        self.assertEqual(
            policy.score(trusted_document("FORUM", "Claim.", "user-generated/forum")),
            0.95,
        )
        strict_policy = SourceReliabilityPolicy({"user-generated/forum": 0.0})
        self.assertEqual(
            strict_policy.score({
                **trusted_document("FORUM", "Claim.", "user-generated/forum"),
                "reliability": "trusted",
            }),
            0.0,
        )

    def test_conflict_without_independent_support_stays_unresolved(self):
        documents = [
            trusted_document("D1", "X increases Y.", "official/government"),
            trusted_document("D2", "X decreases Y.", "user-generated/forum"),
        ]

        _, conflicts = ContradictionDetectionAgent(
            nli_classifier=self.nli
        ).detect_and_resolve(documents)

        self.assertIsNone(conflicts[0]["preferred_document"])
        self.assertIn("unresolved", conflicts[0]["resolution"])

    def test_hallucination_detector_uses_verified_claims_only(self):
        answer = "Photosynthesis lets plants use sunlight to make sugars."
        evidence = trusted_document(
            "SCI1", "Photosynthesis lets plants use sunlight to make sugars."
        )
        evidence["verification_claims"] = [
            {
                "claim": "Photosynthesis lets plants use sunlight to make sugars.",
                "source_document_ids": ["SCI1"],
                "status": "UNSUPPORTED",
                "evidence": "",
                "confidence": 0.0,
            }
        ]

        result = HallucinationDetectionAgent(nli_classifier=self.nli).check(
            answer, [evidence]
        )

        self.assertEqual(result["claims"][0]["status"], "UNSUPPORTED")
        self.assertEqual(result["supported_ratio"], 0.0)

    def test_hallucination_detector_flags_contradicted_claim(self):
        result = HallucinationDetectionAgent(nli_classifier=self.nli).check(
            "X decreases Y.",
            [trusted_document("D1", "X increases Y.")],
        )

        self.assertEqual(result["claims"][0]["status"], "CONTRADICTED")
        self.assertEqual(result["flags"][0]["issue_type"], "CONTRADICTED_BY_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
