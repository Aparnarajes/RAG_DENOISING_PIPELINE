import unittest

import benchmark


class FixedNLIClassifier:
    def __init__(self, relationship="ENTAILMENT", confidence=0.99):
        self.relationship = relationship
        self.confidence = confidence
        self.pairs = []

    def classify_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [
            {"relationship": self.relationship, "confidence": self.confidence}
            for _ in pairs
        ]


class SequenceNLIClassifier:
    def __init__(self, relationships):
        self.relationships = relationships
        self.pairs = []

    def classify_pairs(self, pairs):
        self.pairs.extend(pairs)
        return self.relationships[:len(pairs)]


class FixedPipeline:
    def __init__(self, result, retrieved_documents):
        self.result = result
        self.calls = []
        self.retrieval_agent = FixedRetrievalAgent(retrieved_documents)

    def run(self, query, top_k):
        self.calls.append((query, top_k))
        return self.result


class FixedRetrievalAgent:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k):
        return self.documents[:top_k]


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_has_five_queries_per_required_category(self):
        documents, queries = benchmark.load_inputs()

        self.assertEqual(len(documents), 22)
        self.assertEqual(len(queries), 20)
        self.assertEqual(
            {category: 5 for category in benchmark.EXPECTED_CATEGORIES},
            benchmark.Counter(query["category"] for query in queries),
        )

    def test_expected_fact_coverage_requires_all_fact_tokens(self):
        result = benchmark.evaluate_expected_facts(
            "Water generally ranges between 90 and 96 degrees Celsius.",
            ["90 and 96 degrees Celsius", "hotter water over-extracts bitter compounds"],
        )

        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["matched_facts"], ["90 and 96 degrees Celsius"])

    def test_unsupported_answer_claim_is_not_counted_as_grounded(self):
        nli = FixedNLIClassifier("NEUTRAL", 0.99)

        result = benchmark.evaluate_answer_grounding(
            "Photosynthesis converts sunlight into stored chemical energy.",
            [],
            nli,
        )

        self.assertEqual(result["claim_count"], 1)
        self.assertEqual(result["unsupported_claim_count"], 1)
        self.assertEqual(result["hallucination_rate"], 1.0)
        self.assertEqual(nli.pairs, [])

    def test_claim_extraction_keeps_plain_factual_sentence(self):
        self.assertEqual(
            benchmark.extract_evaluation_claims(
                "Water boils at 100 degrees Celsius at standard atmospheric pressure."
            ),
            ["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
        )

    def test_conflict_answer_claim_extraction_omits_structured_metadata(self):
        answer = (
            'Synthesized answer for "What happened?":\n'
            "Conflicting claims were detected:\n"
            "- [D1, source priority 0.75] The event happened in 2020. vs. "
            "[D2, source priority 0.75] The event happened in 2021. "
            "(confidence 0.91).\n"
            "  Resolution: Conflict retained as unresolved.\n"
            "Confidence: 0.72\n"
            "• The event happened in 2020. [D1]\n"
        )

        self.assertEqual(
            benchmark.extract_evaluation_claims(answer),
            ["The event happened in 2020.", "The event happened in 2021."],
        )

    def test_claim_extraction_excludes_confidence_and_resolution_fragments(self):
        answer = (
            "## Evaluation summary\n"
            "Resolution: Prefer neither source.\n"
            "Confidence: 0.95\n"
            "The reported result was 12 units. (confidence 0.95)."
        )

        self.assertEqual(
            benchmark.extract_evaluation_claims(answer),
            ["The reported result was 12 units."],
        )

    def test_claim_extraction_keeps_multiple_and_unsupported_facts(self):
        answer = (
            "The device uses solar power. It operates continuously at night."
        )

        self.assertEqual(len(benchmark.extract_evaluation_claims(answer)), 2)
        result = benchmark.evaluate_answer_grounding(
            "The device operates continuously at night.",
            [],
            FixedNLIClassifier("NEUTRAL", 0.99),
        )
        self.assertEqual(result["claims"][0]["status"], "UNSUPPORTED")

    def test_claim_supported_and_contradicted_by_credible_sources_is_conflicted(self):
        documents = [
            {
                "id": "D1",
                "reliability": "trusted",
                "text": "The device is made of steel.",
            },
            {
                "id": "D2",
                "reliability": "trusted",
                "text": "The device is made of aluminum.",
            },
        ]
        nli = SequenceNLIClassifier([
            {"relationship": "ENTAILMENT", "confidence": 0.96},
            {"relationship": "CONTRADICTION", "confidence": 0.94},
        ])

        result = benchmark.evaluate_answer_grounding(
            "The device is made of steel.", documents, nli
        )

        self.assertEqual(result["claims"][0]["status"], "CONFLICTED")
        self.assertTrue(result["claims"][0]["conflicted"])
        self.assertEqual(result["claims"][0]["evidence_document_ids"], ["D1", "D2"])
        self.assertEqual(result["conflicted_claim_count"], 1)
        self.assertGreaterEqual(result["claims"][0]["confidence"], 0.0)
        self.assertLessEqual(result["claims"][0]["confidence"], 1.0)

    def test_abstention_has_no_factual_hallucination_claim(self):
        result = benchmark.evaluate_answer_grounding(
            "Insufficient verified evidence is available to answer this question reliably.",
            [],
            FixedNLIClassifier(),
        )

        self.assertTrue(result["abstained"])
        self.assertEqual(result["claim_count"], 0)
        self.assertEqual(result["hallucination_rate"], 0.0)

    def test_expected_conflict_requires_matching_document_pair(self):
        query = {
            "expected_conflicts": [{"document_ids": ["d1", "d2"]}],
        }

        self.assertEqual(
            benchmark._expected_conflicts_detected(
                query, [{"doc_a": "d2", "doc_b": "d1", "relationship": "CONTRADICTION"}]
            ),
            1,
        )
        self.assertEqual(
            benchmark._expected_conflicts_detected(
                query, [{"doc_a": "d1", "doc_b": "d3", "relationship": "CONTRADICTION"}]
            ),
            0,
        )

    def test_both_systems_receive_identical_query_and_top_k(self):
        document = {"id": "d1", "text": "A trusted source states a concrete fact."}
        result = {
            "retrieved_documents": [document],
            "filtered_documents": [document],
            "final_answer": "A trusted source states a concrete fact.",
            "citations": ["d1"],
            "structured_citations": [],
            "confidence_score": 0.8,
            "conflicts_detected": [],
            "hallucination_check": {},
        }
        standard = FixedPipeline(result, [document])
        denoised = FixedPipeline(result, [document])
        query = {
            "id": "Q-test",
            "category": "normal",
            "query": "What is the concrete fact?",
            "relevant_document_ids": ["d1"],
            "expected_facts": ["trusted source states a concrete fact"],
            "expected_behavior": "answer",
        }

        row = benchmark.evaluate_query(
            query,
            standard,
            denoised,
            [document],
            FixedNLIClassifier(),
        )

        self.assertEqual(standard.calls, [(query["query"], benchmark.TOP_K)])
        self.assertEqual(denoised.calls, [(query["query"], benchmark.TOP_K)])
        self.assertEqual(row["systems"]["standard"]["metrics"]["answer_accuracy"], 1.0)
        self.assertEqual(row["systems"]["denoised"]["metrics"]["answer_accuracy"], 1.0)

    def test_aggregate_reports_supported_abstention_conflict_and_overall_rates(self):
        def system_record(status, abstained=False):
            claim_results = [] if status is None else [{"status": status}]
            return {
                "evidence_confidence_score": 0.7,
                "measured_end_to_end_latency_seconds": 0.1,
                "pipeline_hallucination_check": {
                    "hallucination_rate": 0.25,
                },
                "feedback_loop": {"triggered": False, "retry_count": 0, "max_retries": 1},
                "metrics": {
                    "retrieval_precision": 1.0,
                    "retrieval_recall": 1.0,
                    "noise_documents_before_filtering": 0,
                    "noise_documents_after_filtering": 0,
                    "expected_fact_coverage": 1.0,
                    "answer_accuracy": 1.0,
                    "correct_insufficient_evidence_behavior": abstained,
                    "unreliable_documents_passed_to_generation": 0,
                    "generation_context_characters": 20,
                    "independent_grounding_evaluation": {
                        "claims": claim_results,
                        "claim_count": len(claim_results),
                        "supported_claim_count": int(status == "SUPPORTED"),
                        "unsupported_claim_count": int(status == "UNSUPPORTED"),
                        "contradicted_claim_count": int(status == "CONTRADICTED"),
                        "conflicted_claim_count": int(status == "CONFLICTED"),
                        "unsupported_or_contradicted_claim_count": int(
                            status in {"UNSUPPORTED", "CONTRADICTED"}
                        ),
                    },
                    "matched_expected_facts": ["fact"] if status else [],
                },
            }

        categories = [
            ("normal", "answer", "SUPPORTED", False, 0, 0),
            ("noisy_irrelevant", "answer", "UNSUPPORTED", False, 0, 0),
            ("conflicting", "answer", "CONTRADICTED", False, 1, 1),
            ("unsupported", "insufficient_evidence", None, True, 0, 0),
        ]
        rows = []
        for index, (category, behavior, status, abstained, expected, detected) in enumerate(categories):
            rows.append({
                "id": f"Q{index}",
                "category": category,
                "ground_truth": {
                    "expected_behavior": behavior,
                    "expected_facts": ["fact"] if behavior == "answer" else [],
                },
                "systems": {
                    "standard": system_record(status, abstained),
                    "denoised": system_record(status, abstained),
                },
                "conflict_evaluation": {
                    "expected_conflict_count": expected,
                    "detected_expected_conflict_count": detected,
                },
            })

        aggregates = benchmark._aggregate(rows)

        self.assertEqual(aggregates["denoised"]["supported_query_factual_claim_rate"], 1 / 3)
        self.assertEqual(aggregates["denoised"]["unsupported_query_abstention_rate"], 1.0)
        self.assertEqual(aggregates["denoised"]["contradiction_handling_rate"], 1.0)
        self.assertEqual(aggregates["denoised"]["unsupported_claim_rate"], 1 / 3)
        self.assertEqual(
            aggregates["denoised"]["unsupported_or_contradicted_claim_rate"],
            2 / 3,
        )
        self.assertEqual(aggregates["denoised"]["pipeline_hallucination_rate_mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
