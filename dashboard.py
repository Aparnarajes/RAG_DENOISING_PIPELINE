"""Serve the live Standard RAG vs Denoised RAG engineering dashboard."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse

from benchmark import extract_evaluation_claims
from orchestrator import MAX_QUERY_LENGTH, build_pipelines
from utils.embeddings import normalize_text


ROOT = Path(__file__).parent
UI_PATH = ROOT / "dashboard.html"
SUMMARY_PATH = ROOT / "benchmark" / "results" / "summary.json"
QUERY_PATH = ROOT / "data" / "benchmark_queries.json"
MAX_REQUEST_BODY_BYTES = MAX_QUERY_LENGTH * 8 + 1024
_pipeline_lock = threading.Lock()
_pipelines = None


def run_query_comparison(query: str, pipelines=None) -> dict:
    """Run both real Python pipelines and return their inspectable outputs."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Please enter a question.")
    if len(normalized_query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query must be at most {MAX_QUERY_LENGTH} characters.")

    standard, denoised = pipelines if pipelines is not None else get_pipelines()
    with _pipeline_lock:
        standard_result = standard.run(normalized_query)
        denoised_result = denoised.run(normalized_query)
    return {
        "query": normalized_query,
        "standard": standard_result,
        "denoised": denoised_result,
        "claim_display": {
            "standard": filter_claims_for_display(standard_result),
            "denoised": filter_claims_for_display(denoised_result),
        },
    }


def filter_claims_for_display(result: dict) -> list[dict]:
    """Keep only evaluated claims that match factual assertions in the answer."""
    answer = str(result.get("final_answer", ""))
    check = result.get("hallucination_check")
    if not isinstance(check, dict):
        return []
    evaluated_claims = check.get("claims")
    if not isinstance(evaluated_claims, list):
        return []

    answer_claims = extract_evaluation_claims(answer)
    display_claims = []
    for answer_claim in answer_claims:
        normalized_answer_claim = normalize_text(answer_claim)
        match = next(
            (
                claim for claim in evaluated_claims
                if isinstance(claim, dict)
                and normalized_answer_claim
                and normalized_answer_claim in normalize_text(
                    str(claim.get("claim", claim.get("sentence", "")))
                )
            ),
            None,
        )
        if match is not None:
            display_claims.append({**match, "claim": answer_claim})
    return display_claims


def get_pipelines():
    """Initialize embedding/NLI-backed pipelines once per server process."""
    global _pipelines
    if _pipelines is None:
        with _pipeline_lock:
            if _pipelines is None:
                pipelines = build_pipelines()
                pipelines[1].contradiction_agent.nli.classify_pairs([
                    ("A sample premise is true.", "A sample premise is true."),
                ])
                _pipelines = pipelines
    return _pipelines


def load_benchmark_summary() -> dict:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "Benchmark results are missing. Run `python benchmark.py` first."
        )
    with SUMMARY_PATH.open(encoding="utf-8") as summary_file:
        payload = json.load(summary_file)
    if not isinstance(payload.get("aggregates"), dict):
        raise ValueError("Benchmark summary has no aggregate metrics.")
    return payload


def load_demo_queries() -> list[dict[str, str]]:
    if not QUERY_PATH.is_file():
        raise FileNotFoundError("Benchmark query examples are unavailable.")
    with QUERY_PATH.open(encoding="utf-8") as query_file:
        queries = json.load(query_file)
    example_categories = (
        ("normal", "Supported query"),
        ("noisy_irrelevant", "Noisy query"),
        ("conflicting", "Conflicting query"),
        ("unsupported", "Unsupported query"),
    )
    examples = []
    for category, label in example_categories:
        match = next(
            (item for item in queries if item.get("category") == category),
            None,
        )
        if match:
            examples.append({"label": label, "query": str(match["query"])})
    return examples


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RAGDenoisingDashboard/1.0"

    def _send_json(self, payload: Any, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            try:
                content = UI_PATH.read_bytes()
            except OSError:
                self._send_json({"error": "Dashboard UI is unavailable."}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if route == "/api/benchmark":
            try:
                self._send_json(load_benchmark_summary())
            except FileNotFoundError:
                self._send_json(
                    {"error": "Benchmark results are missing. Run `python benchmark.py` first."},
                    503,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                self._send_json({"error": "Saved benchmark results could not be loaded."}, 503)
            return
        if route == "/api/examples":
            try:
                self._send_json({"examples": load_demo_queries()})
            except (OSError, ValueError, json.JSONDecodeError):
                self._send_json({"error": "Demo query examples could not be loaded."}, 503)
            return
        self._send_json({"error": "Not found."}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self._send_json({"error": "Not found."}, 404)
            return
        try:
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                self._send_json(
                    {"error": "Request Content-Type must be application/json."},
                    400,
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json({"error": "Content-Length must be a valid integer."}, 400)
                return
            if content_length <= 0:
                self._send_json({"error": "Request body must not be empty."}, 400)
                return
            if content_length > MAX_REQUEST_BODY_BYTES:
                self._send_json({"error": "Request body exceeds the allowed size."}, 413)
                return
            try:
                body = json.loads(self.rfile.read(content_length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json({"error": "Request body must contain valid JSON."}, 400)
                return
            if not isinstance(body, dict) or not isinstance(body.get("query"), str):
                self._send_json(
                    {"error": "Request body must contain a string query."},
                    400,
                )
                return
            response = run_query_comparison(body["query"])
        except ValueError as exc:
            message = str(exc)
            if message == "Please enter a question.":
                error_message = message
            elif message.startswith("Query must be at most"):
                error_message = message
            else:
                error_message = "Request validation failed."
            self._send_json({"error": error_message}, 400)
            return
        except Exception as exc:
            self.log_error("RAG pipeline request failed.")
            self._send_json(
                {"error": "The RAG pipeline request failed. Check the server log."},
                500,
            )
            return
        self._send_json(response)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {format_string % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("Loading the embedding and NLI models once for this server process...")
    try:
        get_pipelines()
    except Exception:
        parser.exit(
            1,
            "Dashboard startup failed. Check Python dependencies, configured model "
            "names, and internet access for first-time model downloads.\n",
        )
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"RAG Denoising Dashboard: http://{args.host}:{args.port}")
    print("Pipelines ready. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
