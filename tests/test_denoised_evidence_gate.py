import os
import unittest
from unittest.mock import patch

from agents.answer_generation_agent import AnswerGenerationAgent
from pipeline.denoised_rag import DenoisedRAGPipeline


class FakeIndex:
    def __init__(self, terms=None):
        self.terms = list(terms or [])
        self.expansion_count = 0

    def top_terms_for_text(self, text, n=4):
        if not self.terms:
            return []
        self.expansion_count += 1
        return [self.terms[(self.expansion_count - 1) % len(self.terms)]]


class FakeRetrievalAgent:
    def __init__(self, documents, index=None):
        self.documents = documents
        self.index = index or FakeIndex()
        self.queries = []

    def retrieve(self, query, top_k=6):
        self.queries.append(query)
        documents = (
            self.documents.get(query, self.documents.get("*", []))
            if isinstance(self.documents, dict) else self.documents
        )
        return [dict(document) for document in documents[:top_k]]


class FakeRelevanceAgent:
    threshold = 0.45

    def __init__(self):
        self.queries = []

    def score(self, query, documents):
        self.queries.append(query)
        scored = []
        for document in documents:
            score = float(
                document.get("test_relevance_by_query", {}).get(
                    query, document.get("test_relevance", 0.0)
                )
            )
            scored.append({
                **document,
                "document_id": document.get("id"),
                "score": score,
                "relevance_score": score,
                "decision": "KEEP" if score >= self.threshold else "REMOVE",
                "reasoning": f"test relevance={score:.2f}",
            })
        return scored

    def filter(self, documents):
        return [document for document in documents if document["decision"] == "KEEP"]


class FakeVerificationAgent:
    trust_threshold = 0.5

    def verify(self, documents):
        verified = []
        for document in documents:
            status = document.get("test_verification_status", "VERIFIED")
            claims = document.get(
                "verification_claims",
                [{"status": status, "claim": document["text"]}],
            )
            verified.append({
                **document,
                "trust_score": document.get("test_trust_score", 0.75),
                "verification_status": status,
                "verification_claims": claims,
            })
        return verified

    def filter(self, documents):
        return [
            document for document in documents
            if (
                document["verification_status"] == "VERIFIED"
                and document["trust_score"] >= self.trust_threshold
                and all(
                    claim["status"] == "VERIFIED"
                    for claim in document["verification_claims"]
                )
            )
        ]


class FakeContradictionAgent:
    def __init__(self, conflicts=None):
        self.conflicts = conflicts or []
        self.documents_seen = []

    def detect_and_resolve(self, documents):
        self.documents_seen = list(documents)
        return list(documents), list(self.conflicts)


class RecordingAnswerAgent:
    def __init__(self):
        self.calls = []

    def generate(self, query, documents, is_standard_rag=False, conflicts=None):
        self.calls.append({
            "query": query,
            "documents": list(documents),
            "conflicts": list(conflicts or []),
        })
        return {
            "answer": "Generated from verified evidence.",
            "citations": [document["id"] for document in documents],
            "structured_citations": [],
            "confidence": 0.8,
            "abstained": False,
            "abstention_reason": None,
            "evidence": [document["id"] for document in documents],
        }


class FakeHallucinationAgent:
    def __init__(self):
        self.calls = []

    def check(self, answer, documents):
        self.calls.append((answer, list(documents)))
        return {"flags": [], "supported_ratio": 1.0, "hallucination_rate": 0.0}


def make_pipeline(
    documents,
    *,
    conflicts=None,
    answer_agent=None,
    max_retries=1,
    index=None,
):
    hallucination_agent = FakeHallucinationAgent()
    return DenoisedRAGPipeline(
        retrieval_agent=FakeRetrievalAgent(documents, index=index),
        relevance_agent=FakeRelevanceAgent(),
        verification_agent=FakeVerificationAgent(),
        contradiction_agent=FakeContradictionAgent(conflicts),
        answer_agent=answer_agent or RecordingAnswerAgent(),
        hallucination_agent=hallucination_agent,
        retry_evidence_confidence_threshold=0.65,
        max_retries=max_retries,
    )


def candidate(document_id, text, *, relevance=0.9, status="VERIFIED", trust=0.75):
    return {
        "id": document_id,
        "title": document_id,
        "text": text,
        "raw_score": 0.8,
        "test_relevance": relevance,
        "test_verification_status": status,
        "test_trust_score": trust,
        "reliability": "trusted",
    }


class DenoisedEvidenceGateTests(unittest.TestCase):
    def test_completely_unsupported_query_abstains(self):
        answer_agent = RecordingAnswerAgent()
        pipeline = make_pipeline([], answer_agent=answer_agent)

        result = pipeline.run("A question outside the knowledge base")

        self.assertTrue(result["abstained"])
        self.assertEqual(
            result["abstention_reason"],
            "No sufficiently relevant verified evidence",
        )
        self.assertEqual(result["generation_evidence"], [])
        self.assertEqual(result["hallucination_check"]["flags"], [])
        self.assertEqual(answer_agent.calls, [])

    def test_irrelevant_retrieved_document_abstains(self):
        answer_agent = RecordingAnswerAgent()
        pipeline = make_pipeline([
            candidate("D1", "An unrelated document.", relevance=0.2)
        ], answer_agent=answer_agent)

        result = pipeline.run("A query unrelated to this document")

        self.assertTrue(result["abstained"])
        self.assertEqual(result["filtered_documents"], [])
        self.assertEqual(answer_agent.calls, [])

    def test_relevant_verified_document_reaches_generation(self):
        answer_agent = RecordingAnswerAgent()
        pipeline = make_pipeline([
            candidate("D1", "A relevant supported fact.")
        ], answer_agent=answer_agent)

        result = pipeline.run("What is the supported fact?")

        self.assertFalse(result["abstained"])
        self.assertEqual(result["generation_evidence"], ["D1"])
        self.assertEqual(len(answer_agent.calls), 1)

    def test_conflicting_verified_evidence_reaches_generation_cautiously(self):
        docs = [
            candidate("D1", "X increases Y."),
            candidate("D2", "X decreases Y."),
        ]
        conflict = {
            "claim_a": "X increases Y.",
            "claim_b": "X decreases Y.",
            "document_a": "D1",
            "document_b": "D2",
            "relationship": "CONTRADICTION",
            "confidence": 0.96,
            "source_priority_a": 0.9,
            "source_priority_b": 0.8,
            "resolution": "Conflict retained as unresolved.",
        }
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }):
            answer_agent = AnswerGenerationAgent()
        pipeline = make_pipeline(docs, conflicts=[conflict], answer_agent=answer_agent)

        result = pipeline.run("What do sources say about X and Y?")

        self.assertFalse(result["abstained"])
        self.assertEqual(set(result["generation_evidence"]), {"D1", "D2"})
        self.assertIn("Conflicting claims were detected", result["final_answer"])
        self.assertIn("X increases Y.", result["final_answer"])
        self.assertIn("X decreases Y.", result["final_answer"])
        self.assertIn("disputed, not as settled facts", result["final_answer"])

    def test_only_unsupported_or_contradicted_evidence_abstains(self):
        for status in ("UNSUPPORTED", "CONTRADICTED"):
            with self.subTest(status=status):
                answer_agent = RecordingAnswerAgent()
                pipeline = make_pipeline([
                    candidate("D1", "A claim without verified support.", status=status)
                ], answer_agent=answer_agent)

                result = pipeline.run("What is the claim?")

                self.assertTrue(result["abstained"])
                self.assertEqual(result["generation_evidence"], [])
                self.assertEqual(answer_agent.calls, [])

    def test_retry_documents_are_still_gated_by_original_question(self):
        original_query = "Original unsupported question"
        expanded_query = f"{original_query} added"
        retry_document = candidate(
            "D1", "A document found only by expanded retrieval.", relevance=0.9
        )
        retry_document["test_relevance_by_query"] = {
            original_query: 0.2,
            expanded_query: 0.9,
        }
        documents = {
            original_query: [
                candidate("D0", "An irrelevant initial result.", relevance=0.1)
            ],
            expanded_query: [retry_document],
        }
        pipeline = make_pipeline(documents, index=FakeIndex(["added"]))

        result = pipeline.run(original_query)

        self.assertTrue(result["abstained"])
        self.assertEqual(result["filtered_documents"], [])
        self.assertIn(original_query, pipeline.relevance_agent.queries)
        self.assertNotIn(expanded_query, pipeline.relevance_agent.queries)
        self.assertEqual(pipeline.retrieval_agent.queries[-1], expanded_query)

    def test_maximum_retry_count_is_respected(self):
        pipeline = make_pipeline(
            [candidate("D1", "An irrelevant document.", relevance=0.1)],
            index=FakeIndex(["expanded", "terms"]),
            max_retries=1,
        )

        result = pipeline.run("An unsupported question")

        self.assertEqual(result["feedback_loop"]["retry_count"], 1)
        self.assertEqual(result["feedback_loop"]["max_retries"], 1)
        self.assertEqual(len(pipeline.retrieval_agent.queries), 2)
        self.assertTrue(result["abstained"])

    def test_zero_max_retries_disables_retry(self):
        pipeline = make_pipeline(
            [candidate("D1", "An irrelevant document.", relevance=0.1)],
            index=FakeIndex(["expanded"]),
            max_retries=0,
        )

        result = pipeline.run("An unsupported question")

        self.assertEqual(result["feedback_loop"]["retry_count"], 0)
        self.assertEqual(len(pipeline.retrieval_agent.queries), 1)

    def test_negative_max_retries_are_rejected(self):
        with self.assertRaises(ValueError):
            make_pipeline([], max_retries=-1)

    def test_retry_limit_reads_environment_configuration(self):
        with patch.dict(os.environ, {"RAG_MAX_RETRIES": "2"}):
            pipeline = make_pipeline([], max_retries=None)

        self.assertEqual(pipeline.max_retries, 2)

    def test_invalid_retry_limit_configuration_is_reported(self):
        with patch.dict(os.environ, {"RAG_MAX_RETRIES": "many"}):
            with self.assertRaisesRegex(ValueError, "RAG_MAX_RETRIES"):
                make_pipeline([], max_retries=None)


if __name__ == "__main__":
    unittest.main()
