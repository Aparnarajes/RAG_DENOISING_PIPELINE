from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import dashboard


class FakePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        return self.result


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def post_raw(self, payload, content_type="application/json"):
        request = Request(
            f"{self.base_url}/api/run",
            data=payload,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_live_comparison_calls_both_python_pipelines(self):
        standard_result = {"pipeline_type": "standard_rag"}
        denoised_result = {"pipeline_type": "denoised_rag"}
        standard = FakePipeline(standard_result)
        denoised = FakePipeline(denoised_result)

        result = dashboard.run_query_comparison(
            "  How long does fast charging take?  ",
            pipelines=(standard, denoised),
        )

        self.assertEqual(result["query"], "How long does fast charging take?")
        self.assertIs(result["standard"], standard_result)
        self.assertIs(result["denoised"], denoised_result)
        self.assertEqual(standard.calls, ["How long does fast charging take?"])
        self.assertEqual(denoised.calls, ["How long does fast charging take?"])

    def test_live_comparison_rejects_empty_and_oversized_queries(self):
        standard = FakePipeline({})
        denoised = FakePipeline({})

        with self.assertRaisesRegex(ValueError, "Please enter a question"):
            dashboard.run_query_comparison("  ", pipelines=(standard, denoised))
        with self.assertRaisesRegex(ValueError, "at most"):
            dashboard.run_query_comparison(
                "q" * (dashboard.MAX_QUERY_LENGTH + 1),
                pipelines=(standard, denoised),
            )
        self.assertEqual(standard.calls, [])
        self.assertEqual(denoised.calls, [])

    def test_dashboard_claim_display_excludes_conflict_metadata_and_keeps_evidence(self):
        claim_a = "The event happened in 2020."
        claim_b = "The event happened in 2021."
        answer = (
            'Synthesized answer for "What happened?":\n'
            "Conflicting claims were detected:\n"
            f"- [D1, source priority 0.75] {claim_a} vs. "
            f"[D2, source priority 0.75] {claim_b} (confidence 0.91).\n"
            "  Resolution: Conflict retained as unresolved.\n"
            "Confidence: 0.72\n"
            f"• {claim_a} [D1]\n"
        )
        raw_claims = [
            {
                "claim": f"[D1, source priority 0.75] {claim_a}",
                "status": "SUPPORTED",
                "confidence": 0.98,
                "evidence_document_ids": ["D1"],
                "evidence": [{"document_id": "D1", "relationship": "ENTAILMENT"}],
            },
            {
                "claim": f"[D2, source priority 0.75] {claim_b}",
                "status": "CONTRADICTED",
                "confidence": 0.97,
                "evidence_document_ids": ["D3"],
                "evidence": [{"document_id": "D3", "relationship": "CONTRADICTION"}],
            },
            {"claim": "(confidence 0.91).", "status": "UNSUPPORTED"},
            {
                "claim": "Resolution: Conflict retained as unresolved.",
                "status": "UNSUPPORTED",
            },
            {"claim": "Confidence: 0.72", "status": "UNSUPPORTED"},
        ]

        display_claims = dashboard.filter_claims_for_display({
            "final_answer": answer,
            "hallucination_check": {"claims": raw_claims},
        })

        self.assertEqual([claim["claim"] for claim in display_claims], [claim_a, claim_b])
        self.assertEqual(
            [claim["status"] for claim in display_claims],
            ["SUPPORTED", "CONTRADICTED"],
        )
        self.assertEqual(display_claims[0]["evidence_document_ids"], ["D1"])
        self.assertEqual(display_claims[0]["evidence"][0]["relationship"], "ENTAILMENT")
        self.assertEqual(display_claims[1]["evidence_document_ids"], ["D3"])
        self.assertEqual(display_claims[1]["evidence"][0]["relationship"], "CONTRADICTION")
        self.assertEqual(len(raw_claims), 5)

    def test_malformed_dashboard_json_returns_safe_client_error(self):
        status, payload = self.post_raw(b'{"query":')

        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "Request body must contain valid JSON."})
        self.assertNotIn("Traceback", json.dumps(payload))

    def test_empty_dashboard_query_returns_actionable_validation_message(self):
        status, payload = self.post_raw(b'{"query":"   "}')

        self.assertEqual(status, 400)
        self.assertEqual(payload, {"error": "Please enter a question."})

    def test_oversized_dashboard_body_is_rejected_without_processing(self):
        body = b" " * (dashboard.MAX_REQUEST_BODY_BYTES + 1)
        status, payload = self.post_raw(body)

        self.assertEqual(status, 413)
        self.assertEqual(payload, {"error": "Request body exceeds the allowed size."})

    def test_dashboard_rejects_non_json_content_type(self):
        status, payload = self.post_raw(b'{"query":"hello"}', "text/plain")

        self.assertEqual(status, 400)
        self.assertEqual(
            payload,
            {"error": "Request Content-Type must be application/json."},
        )

    def test_pipeline_failure_response_does_not_expose_exception_text(self):
        class FailingPipeline:
            def run(self, query):
                raise RuntimeError("provider_secret_value")

        with patch.object(
            dashboard, "get_pipelines", return_value=(FailingPipeline(), FailingPipeline())
        ):
            status, payload = self.post_raw(b'{"query":"A valid question?"}')

        self.assertEqual(status, 500)
        self.assertEqual(
            payload,
            {"error": "The RAG pipeline request failed. Check the server log."},
        )
        self.assertNotIn("provider_secret_value", json.dumps(payload))

    def test_benchmark_summary_is_loaded_from_saved_results(self):
        payload = dashboard.load_benchmark_summary()

        self.assertEqual(payload["metadata"]["query_count"], 20)
        self.assertEqual(payload["aggregates"]["denoised"]["expected_fact_hits"], 30)
        self.assertEqual(
            payload["aggregates"]["denoised"]["correct_insufficient_evidence_count"],
            5,
        )

    def test_demo_queries_include_all_four_existing_categories(self):
        examples = dashboard.load_demo_queries()

        self.assertEqual(
            [example["label"] for example in examples],
            [
                "Supported query",
                "Noisy query",
                "Conflicting query",
                "Unsupported query",
            ],
        )
        self.assertTrue(all(example["query"] for example in examples))

    def test_dashboard_ui_uses_live_api_and_precise_metric_labels(self):
        html = dashboard.UI_PATH.read_text(encoding="utf-8")

        self.assertIn('fetch("/api/run"', html)
        self.assertIn('fetch("/api/benchmark")', html)
        self.assertIn("Expected-Fact Coverage / Abstention Accuracy", html)
        self.assertIn("Evidence Confidence", html)
        self.assertIn("payload.claim_display?.denoised", html)
        self.assertNotIn("simulateQueryResult", html)

    def test_environment_template_uses_blank_credentials_and_ignores_real_env_files(self):
        template = (dashboard.ROOT / ".env.example").read_text(encoding="utf-8")
        gitignore = (dashboard.ROOT / ".gitignore").read_text(encoding="utf-8")

        for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            assignment = next(
                line for line in template.splitlines()
                if line.startswith(f"{name}=")
            )
            self.assertEqual(assignment, f"{name}=")
        self.assertIn(".env\n", gitignore)
        self.assertIn(".env.*\n", gitignore)
        self.assertIn("!.env.example\n", gitignore)
        self.assertNotIn("benchmark/results", gitignore)


if __name__ == "__main__":
    unittest.main()
