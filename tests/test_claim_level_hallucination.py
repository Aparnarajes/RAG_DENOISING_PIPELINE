import unittest
from unittest.mock import patch

from agents.answer_generation_agent import AnswerGenerationAgent
from agents.hallucination_detection_agent import HallucinationDetectionAgent


class ScriptedNLIClassifier:
    def __init__(self, relationships):
        self.relationships = list(relationships)
        self.pairs = []

    def classify_pairs(self, pairs):
        self.pairs.extend(pairs)
        if len(self.relationships) != len(pairs):
            raise AssertionError(
                f"Expected {len(self.relationships)} comparisons, got {len(pairs)}."
            )
        return self.relationships


def relation(label, confidence, probabilities=None):
    return {
        "relationship": label,
        "confidence": confidence,
        "probabilities": probabilities or {label: confidence},
    }


def verified_document(document_id, text):
    return {
        "id": document_id,
        "reliability": "trusted",
        "verification_status": "VERIFIED",
        "verification_claims": [{"claim": text, "status": "VERIFIED"}],
    }


class ClaimLevelHallucinationTests(unittest.TestCase):
    def test_supported_claim_includes_evidence_ids_and_nli_probabilities(self):
        classifier = ScriptedNLIClassifier([
            relation("ENTAILMENT", 0.91, {
                "ENTAILMENT": 0.91,
                "CONTRADICTION": 0.03,
                "NEUTRAL": 0.06,
            })
        ])
        detector = HallucinationDetectionAgent(nli_classifier=classifier)

        result = detector.check(
            "Plants convert sunlight into chemical energy.",
            [verified_document("D4", "Plants convert sunlight into chemical energy.")],
        )

        claim = result["claims"][0]
        self.assertEqual(claim["status"], "SUPPORTED")
        self.assertEqual(claim["evidence_document_ids"], ["D4"])
        self.assertAlmostEqual(claim["confidence"], 0.91)
        self.assertEqual(claim["nli_probabilities"]["ENTAILMENT"], 0.91)

    def test_unsupported_claim_has_no_evidence_ids_and_neutral_confidence(self):
        classifier = ScriptedNLIClassifier([
            relation("NEUTRAL", 0.88, {
                "ENTAILMENT": 0.04,
                "CONTRADICTION": 0.08,
                "NEUTRAL": 0.88,
            })
        ])
        detector = HallucinationDetectionAgent(nli_classifier=classifier)

        result = detector.check(
            "A hidden city exists beneath Saturn's clouds.",
            [verified_document("D4", "Plants convert sunlight into chemical energy.")],
        )

        claim = result["claims"][0]
        self.assertEqual(claim["status"], "UNSUPPORTED")
        self.assertEqual(claim["evidence_document_ids"], [])
        self.assertAlmostEqual(claim["confidence"], 0.88)
        self.assertEqual(claim["nli_probabilities"]["NEUTRAL"], 0.88)

    def test_contradicted_claim_lists_opposing_evidence_ids(self):
        classifier = ScriptedNLIClassifier([
            relation("CONTRADICTION", 0.94),
            relation("CONTRADICTION", 0.91),
        ])
        detector = HallucinationDetectionAgent(nli_classifier=classifier)

        result = detector.check(
            "X increases Y.",
            [
                verified_document("D7", "X decreases Y."),
                verified_document("D9", "X decreases Y."),
            ],
        )

        claim = result["claims"][0]
        self.assertEqual(claim["status"], "CONTRADICTED")
        self.assertEqual(claim["evidence_document_ids"], ["D7", "D9"])
        self.assertAlmostEqual(claim["confidence"], 0.94)

    def test_multiple_claims_can_have_mixed_statuses(self):
        classifier = ScriptedNLIClassifier([
            relation("ENTAILMENT", 0.93),
            relation("NEUTRAL", 0.87),
        ])
        detector = HallucinationDetectionAgent(nli_classifier=classifier)

        result = detector.check(
            "Plants convert sunlight into chemical energy. "
            "A hidden city exists beneath Saturn's clouds.",
            [verified_document("D4", "Plants convert sunlight into chemical energy.")],
        )

        self.assertEqual(
            [claim["status"] for claim in result["claims"]],
            ["SUPPORTED", "UNSUPPORTED"],
        )
        self.assertEqual(result["claims"][0]["evidence_document_ids"], ["D4"])
        self.assertEqual(result["claims"][1]["evidence_document_ids"], [])

    def test_answer_without_factual_claims_returns_empty_claim_list(self):
        classifier = ScriptedNLIClassifier([])
        detector = HallucinationDetectionAgent(nli_classifier=classifier)

        result = detector.check(
            "Thanks!",
            [verified_document("D4", "Plants convert sunlight into chemical energy.")],
        )

        self.assertEqual(result["claims"], [])
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["supported_ratio"], 1.0)
        self.assertEqual(classifier.pairs, [])

    def test_claim_confidence_is_bounded(self):
        classifier = ScriptedNLIClassifier([relation("ENTAILMENT", 1.0)])
        detector = HallucinationDetectionAgent(nli_classifier=classifier)

        result = detector.check(
            "Plants convert sunlight into chemical energy.",
            [verified_document("D4", "Plants convert sunlight into chemical energy.")],
        )

        confidence = result["claims"][0]["confidence"]
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_answer_generation_exposes_bounded_evidence_confidence(self):
        document = {
            "id": "D4",
            "text": "Plants convert sunlight into chemical energy.",
            "reliability": "trusted",
            "trust_score": 0.9,
            "relevance_score": 0.8,
        }
        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }):
            result = AnswerGenerationAgent().generate("What do plants do?", [document])

        self.assertEqual(
            result["evidence_confidence_score"],
            result["evidence_confidence"],
        )
        self.assertGreaterEqual(result["evidence_confidence_score"], 0.0)
        self.assertLessEqual(result["evidence_confidence_score"], 1.0)
        self.assertEqual(result["confidence"], result["evidence_confidence_score"])

    def test_missing_optional_provider_keys_use_local_generation(self):
        document = {
            "id": "D4",
            "text": "Plants convert sunlight into chemical energy.",
            "reliability": "trusted",
            "trust_score": 0.9,
            "relevance_score": 0.8,
        }
        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }):
            result = AnswerGenerationAgent().generate(
                "What do plants do?", [document]
            )

        self.assertIn("Plants convert sunlight", result["answer"])
        self.assertFalse(result["abstained"])


if __name__ == "__main__":
    unittest.main()
