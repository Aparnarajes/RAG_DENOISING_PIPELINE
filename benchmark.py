"""Reproducible 20-query comparison of Standard and Denoised RAG."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any, Sequence
from unittest.mock import patch

from orchestrator import build_pipelines
from utils.evidence_analysis import (
    SourceReliabilityPolicy,
    extract_claims,
)
from utils.embeddings import normalize_text


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUERY_PATH = DATA_DIR / "benchmark_queries.json"
DOCUMENT_PATH = DATA_DIR / "documents.json"
RESULTS_DIR = ROOT / "benchmark" / "results"
TOP_K = 6
EXPECTED_CATEGORIES = ("normal", "noisy_irrelevant", "conflicting", "unsupported")
CLAIM_CONFIDENCE_THRESHOLD = 0.65
ABSTENTION_PHRASES = (
    "insufficient verified evidence",
    "insufficient evidence",
    "not enough information",
    "cannot answer reliably",
    "no reliable evidence",
)
CONFLICT_ENTRY_RE = re.compile(
    r"^\s*[-*]\s*\[(?P<source_a>[^\]]+)\]\s*"
    r"(?P<claim_a>.*?)\s+vs\.\s+"
    r"\[(?P<source_b>[^\]]+)\]\s*(?P<claim_b>.*?)"
    r"(?:\s+\(confidence\s+[\d.]+\)\.?)?\s*$",
    re.IGNORECASE,
)
METADATA_LINE_RE = re.compile(
    r"^\s*(?:resolution|confidence|source\s+priority|status|relationship)\s*:",
    re.IGNORECASE,
)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
MARKDOWN_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")


def extract_evaluation_claims(answer: str) -> list[str]:
    """Extract assertions while excluding pipeline preambles and formatted metadata."""
    claims = []
    seen = set()
    for line in answer.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or METADATA_LINE_RE.match(stripped)
            or MARKDOWN_HEADING_RE.match(stripped)
            or MARKDOWN_RULE_RE.match(stripped)
            or stripped.startswith("```")
            or (stripped.startswith("|") and stripped.endswith("|"))
        ):
            continue

        conflict_entry = CONFLICT_ENTRY_RE.match(stripped)
        if conflict_entry:
            candidates = (
                conflict_entry.group("claim_a").strip(),
                conflict_entry.group("claim_b").strip(),
            )
            for claim in candidates:
                normalized = normalize_text(claim)
                if len(claim.split()) >= 2 and normalized not in seen:
                    claims.append(claim)
                    seen.add(normalized)
            continue

        candidate = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", stripped).strip()
        candidate = re.sub(r"\s*\[[^\]]+\]\s*$", "", candidate).strip()
        candidate = re.sub(
            r"\s*\(confidence\s+[\d.]+\)\.?\s*$", "", candidate, flags=re.IGNORECASE
        ).strip()
        if not candidate or candidate.endswith(":"):
            continue
        if any(phrase in normalize_text(candidate) for phrase in ABSTENTION_PHRASES):
            continue
        for claim in extract_claims(candidate):
            normalized = normalize_text(claim)
            if normalized not in seen:
                claims.append(claim)
                seen.add(normalized)
    return claims


def load_inputs() -> tuple[list[dict], list[dict]]:
    documents = json.loads(DOCUMENT_PATH.read_text())
    queries = json.loads(QUERY_PATH.read_text())
    validate_benchmark_data(queries, documents)
    return documents, queries


def validate_benchmark_data(queries: list[dict], documents: list[dict]) -> None:
    if len(queries) != 20:
        raise ValueError(f"Benchmark must contain exactly 20 queries, found {len(queries)}.")
    category_counts = Counter(query.get("category") for query in queries)
    if category_counts != Counter({category: 5 for category in EXPECTED_CATEGORIES}):
        raise ValueError(
            "Benchmark must contain exactly five queries in each category: "
            + ", ".join(EXPECTED_CATEGORIES)
        )

    document_ids = {str(document["id"]) for document in documents}
    query_ids = [query.get("id") for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Benchmark query IDs must be unique.")

    for query in queries:
        if not query.get("query", "").strip():
            raise ValueError(f"{query.get('id')} has an empty query.")
        relevant_ids = query.get("relevant_document_ids")
        facts = query.get("expected_facts")
        if not isinstance(relevant_ids, list) or not isinstance(facts, list):
            raise ValueError(f"{query['id']} must define relevant IDs and expected facts as lists.")
        if not set(relevant_ids) <= document_ids:
            raise ValueError(f"{query['id']} references document IDs absent from the corpus.")
        if query["expected_behavior"] == "insufficient_evidence" and (
            relevant_ids or facts
        ):
            raise ValueError(
                f"{query['id']} must not contain evidence or expected facts when unsupported."
            )
        source_tokens = {
            token
            for document in documents
            if (
                document["id"] in relevant_ids
                and str(document.get("reliability", "")).lower() == "trusted"
            )
            for token in normalize_text(document.get("text", "")).split()
        }
        for fact in facts:
            fact_tokens = set(normalize_text(fact).split())
            if not fact_tokens or not fact_tokens <= source_tokens:
                raise ValueError(
                    f"Expected fact in {query['id']} is not grounded in its relevant corpus documents: {fact}"
                )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _document_ids(documents: Sequence[dict]) -> list[str]:
    return [str(document.get("id", document.get("document_id", ""))) for document in documents]


def _answer_claims(answer: str) -> list[str]:
    return extract_evaluation_claims(answer)


def _is_abstention(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def evaluate_expected_facts(answer: str, expected_facts: Sequence[str]) -> dict:
    """Exact token coverage for source-derived facts; no embedding or pipeline score."""
    answer_tokens = set(normalize_text(answer).split())
    matched = [
        fact for fact in expected_facts
        if set(normalize_text(fact).split()) <= answer_tokens
    ]
    return {
        "matched_facts": matched,
        "missing_facts": [fact for fact in expected_facts if fact not in matched],
        "coverage": _safe_ratio(len(matched), len(expected_facts)),
    }


def evaluate_answer_grounding(
    answer: str,
    relevant_documents: Sequence[dict],
    nli_classifier,
) -> dict:
    """Independently classify answer claims against benchmark-ground-truth evidence."""
    claims = _answer_claims(answer)
    source_policy = SourceReliabilityPolicy()
    evidence = [
        {
            "document_id": str(document["id"]),
            "claim": claim,
            "source_priority": source_policy.score(document),
        }
        for document in relevant_documents
        for claim in extract_claims(str(document.get("text", "")))
    ]
    pairs = [
        (evidence_item["claim"], answer_claim)
        for answer_claim in claims
        for evidence_item in evidence
    ]
    relationships = nli_classifier.classify_pairs(pairs) if pairs else []
    if len(relationships) != len(pairs):
        raise ValueError("NLI evaluator returned an unexpected number of claim comparisons.")

    claim_results = []
    evidence_count = len(evidence)
    for claim_index, answer_claim in enumerate(claims):
        matches = []
        for evidence_index, evidence_item in enumerate(evidence):
            relation = relationships[claim_index * evidence_count + evidence_index]
            if float(relation["confidence"]) >= CLAIM_CONFIDENCE_THRESHOLD:
                matches.append({
                    "document_id": evidence_item["document_id"],
                    "evidence": evidence_item["claim"],
                    "source_priority": evidence_item["source_priority"],
                    "relationship": relation["relationship"],
                    "confidence": float(relation["confidence"]),
                })
        credible_matches = [
            match for match in matches
            if match["source_priority"] >= source_policy.min_trusted_score
        ]
        supporting = [
            match for match in credible_matches
            if match["relationship"] == "ENTAILMENT"
        ]
        contradicting = [
            match for match in credible_matches
            if match["relationship"] == "CONTRADICTION"
        ]
        if supporting and contradicting:
            status = "CONFLICTED"
        elif supporting:
            status = "SUPPORTED"
        elif contradicting:
            status = "CONTRADICTED"
        else:
            status = "UNSUPPORTED"
        relevant_matches = supporting + contradicting
        confidence = max(
            (match["confidence"] for match in relevant_matches),
            default=max(
                (match["confidence"] for match in matches),
                default=0.0,
            ),
        )
        claim_results.append({
            "claim": answer_claim,
            "status": status,
            "conflicted": bool(supporting and contradicting),
            "confidence": confidence,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "evidence_document_ids": list(dict.fromkeys(
                match["document_id"] for match in relevant_matches
            )),
            "nli_comparisons": matches,
        })

    unsupported_count = sum(
        result["status"] == "UNSUPPORTED" for result in claim_results
    )
    contradicted_count = sum(
        result["status"] == "CONTRADICTED" for result in claim_results
    )
    conflicted_count = sum(
        result["status"] == "CONFLICTED" for result in claim_results
    )
    unsupported_or_contradicted_count = (
        unsupported_count + contradicted_count + conflicted_count
    )
    return {
        "claims": claim_results,
        "claim_count": len(claim_results),
        "supported_claim_count": sum(
            result["status"] == "SUPPORTED" for result in claim_results
        ),
        "unsupported_claim_count": unsupported_count,
        "contradicted_claim_count": contradicted_count,
        "conflicted_claim_count": conflicted_count,
        "unsupported_or_contradicted_claim_count": unsupported_or_contradicted_count,
        "supported_claim_rate": _safe_ratio(
            sum(result["status"] == "SUPPORTED" for result in claim_results),
            len(claim_results),
        ),
        "unsupported_claim_rate": _safe_ratio(unsupported_count, len(claim_results)),
        "contradicted_claim_rate": _safe_ratio(contradicted_count, len(claim_results)),
        "conflicted_claim_rate": _safe_ratio(conflicted_count, len(claim_results)),
        "unsupported_or_contradicted_claim_rate": _safe_ratio(
            unsupported_or_contradicted_count, len(claim_results)
        ),
        # Compatibility alias; explicitly documents the combined population.
        "hallucination_rate": _safe_ratio(
            unsupported_or_contradicted_count, len(claim_results)
        ),
        "abstained": _is_abstention(answer),
    }


def _pipeline_record(result: dict, measured_seconds: float, system: str) -> dict:
    retrieved = result.get("retrieved_documents", [])
    filtered = result.get("filtered_documents", [])
    return {
        "system": system,
        "retrieved_document_ids": _document_ids(retrieved),
        "retrieved_documents": retrieved,
        "relevance_scored_documents": result.get("relevance_scored_documents", []),
        "relevance_scores": result.get("relevance_scores", []),
        "evidence_verified_documents": result.get("evidence_verified_documents", []),
        "conflicts_detected": result.get("conflicts_detected", []),
        "filtered_document_ids": _document_ids(filtered),
        "filtered_documents": filtered,
        "generated_answer": result.get("final_answer", ""),
        "abstained": result.get("abstained", False),
        "abstention_reason": result.get("abstention_reason"),
        "generation_evidence": result.get("generation_evidence", _document_ids(filtered)),
        "citations": result.get("citations", []),
        "structured_citations": result.get("structured_citations", []),
        "pipeline_hallucination_check": result.get("hallucination_check", {}),
        "evidence_confidence_score": result.get(
            "evidence_confidence_score",
            result.get("confidence_score", result.get("confidence")),
        ),
        "feedback_loop": result.get("feedback_loop", {}),
        "reported_pipeline_latency_seconds": result.get("latency_seconds"),
        "measured_end_to_end_latency_seconds": measured_seconds,
    }


def _expected_conflicts_detected(query: dict, conflicts: Sequence[dict]) -> int:
    detected_pairs = set()
    for conflict in conflicts:
        if conflict.get("relationship") != "CONTRADICTION":
            continue
        pair = {
            str(conflict.get("doc_a", conflict.get("document_a", ""))),
            str(conflict.get("doc_b", conflict.get("document_b", ""))),
        }
        if len(pair) == 2:
            detected_pairs.add(frozenset(pair))
    return sum(
        frozenset(conflict["document_ids"]) in detected_pairs
        for conflict in query.get("expected_conflicts", [])
    )


def evaluate_query(
    query: dict,
    standard_pipeline,
    denoised_pipeline,
    documents: list[dict],
    nli_classifier,
) -> dict:
    query_text = query["query"]
    # The denoised pipeline may rerun retrieval with an expanded feedback query.
    # Keep the initial, identical query retrieval separate for apples-to-apples recall.
    shared_initial_retrieval = standard_pipeline.retrieval_agent.retrieve(
        query_text, top_k=TOP_K
    )
    shared_initial_ids = set(_document_ids(shared_initial_retrieval))
    start = time.perf_counter()
    standard_result = standard_pipeline.run(query_text, top_k=TOP_K)
    standard_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    denoised_result = denoised_pipeline.run(query_text, top_k=TOP_K)
    denoised_elapsed = time.perf_counter() - start

    document_by_id = {str(document["id"]): document for document in documents}
    relevant_ids = set(query["relevant_document_ids"])
    relevant_documents = [
        document_by_id[document_id]
        for document_id in query["relevant_document_ids"]
    ]
    system_records = {}
    for name, result, elapsed in (
        ("standard", standard_result, standard_elapsed),
        ("denoised", denoised_result, denoised_elapsed),
    ):
        record = _pipeline_record(result, elapsed, name)
        retrieved_ids = shared_initial_ids
        passed_ids = set(record["filtered_document_ids"])
        record["shared_initial_retrieval_documents"] = shared_initial_retrieval
        record["shared_initial_retrieval_document_ids"] = _document_ids(
            shared_initial_retrieval
        )
        precision = _safe_ratio(len(retrieved_ids & relevant_ids), len(retrieved_ids))
        recall = (
            len(retrieved_ids & relevant_ids) / len(relevant_ids)
            if relevant_ids else None
        )
        noise_before = len(retrieved_ids - relevant_ids)
        noise_after = len(passed_ids - relevant_ids)
        answer = record["generated_answer"]
        fact_result = evaluate_expected_facts(answer, query["expected_facts"])
        grounding = evaluate_answer_grounding(answer, relevant_documents, nli_classifier)
        if query["expected_behavior"] == "insufficient_evidence":
            correct_behavior = record["abstained"] and grounding["abstained"]
            evaluation_score = float(correct_behavior)
        else:
            correct_behavior = None
            evaluation_score = fact_result["coverage"]
        record["metrics"] = {
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "noise_documents_before_filtering": noise_before,
            "noise_documents_after_filtering": noise_after,
            "noise_reduction": _safe_ratio(noise_before - noise_after, noise_before),
            "expected_fact_coverage": fact_result["coverage"],
            "matched_expected_facts": fact_result["matched_facts"],
            "missing_expected_facts": fact_result["missing_facts"],
            "expected_fact_coverage_abstention_accuracy": evaluation_score,
            "answer_accuracy": evaluation_score,
            "correct_insufficient_evidence_behavior": correct_behavior,
            "independent_grounding_evaluation": grounding,
            "unreliable_documents_passed_to_generation": sum(
                document.get("reliability") != "trusted"
                for document in record["filtered_documents"]
            ),
            "generation_context_characters": sum(
                len(str(document.get("text", ""))) for document in record["filtered_documents"]
            ),
        }
        system_records[name] = record

    conflicts = denoised_result.get("conflicts_detected", [])
    return {
        "id": query["id"],
        "category": query["category"],
        "query": query_text,
        "ground_truth": {
            "relevant_document_ids": query["relevant_document_ids"],
            "expected_facts": query["expected_facts"],
            "expected_behavior": query["expected_behavior"],
            "expected_conflicts": query.get("expected_conflicts", []),
        },
        "shared_initial_retrieval_document_ids": _document_ids(shared_initial_retrieval),
        "systems": system_records,
        "conflict_evaluation": {
            "expected_conflict_count": len(query.get("expected_conflicts", [])),
            "detected_expected_conflict_count": _expected_conflicts_detected(
                query, conflicts
            ),
        },
    }


def _latency_statistics(samples: Sequence[float]) -> dict:
    ordered = sorted(samples)
    if not ordered:
        return {key: 0.0 for key in ("mean", "median", "p95", "min", "max")}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _aggregate(rows: list[dict]) -> dict:
    aggregates = {}
    for system_name in ("standard", "denoised"):
        records = [row["systems"][system_name] for row in rows]
        metric_rows = [record["metrics"] for record in records]
        recalls = [
            metric["retrieval_recall"] for metric in metric_rows
            if metric["retrieval_recall"] is not None
        ]
        total_noise_before = sum(
            metric["noise_documents_before_filtering"] for metric in metric_rows
        )
        total_noise_after = sum(
            metric["noise_documents_after_filtering"] for metric in metric_rows
        )
        total_claims = sum(
            metric["independent_grounding_evaluation"]["claim_count"]
            for metric in metric_rows
        )
        unsupported_claims = sum(
            metric["independent_grounding_evaluation"]["unsupported_claim_count"]
            for metric in metric_rows
        )
        contradicted_claims = sum(
            metric["independent_grounding_evaluation"].get(
                "contradicted_claim_count", 0
            )
            for metric in metric_rows
        )
        conflicted_claims = sum(
            metric["independent_grounding_evaluation"].get(
                "conflicted_claim_count", 0
            )
            for metric in metric_rows
        )
        supported_claims = sum(
            metric["independent_grounding_evaluation"].get(
                "supported_claim_count",
                sum(
                    claim["status"] == "SUPPORTED"
                    for claim in metric["independent_grounding_evaluation"]["claims"]
                ),
            )
            for metric in metric_rows
        )
        fact_coverage_values = [
            metric["expected_fact_coverage"]
            for row, metric in zip(rows, metric_rows)
            if row["ground_truth"]["expected_facts"]
        ]
        insufficient_evidence_count = sum(
            row["ground_truth"]["expected_behavior"] == "insufficient_evidence"
            for row in rows
        )
        correct_insufficient_evidence_count = sum(
            metric["correct_insufficient_evidence_behavior"] is True
            for metric in metric_rows
        )
        expected_fact_hits = sum(
            len(metric["matched_expected_facts"])
            for metric in metric_rows
        )
        expected_fact_total = sum(
            len(row["ground_truth"]["expected_facts"])
            for row in rows
        )
        unsupported_or_contradicted_claims = unsupported_claims + contradicted_claims
        aggregates[system_name] = {
            "retrieval_precision_mean": statistics.mean(
                metric["retrieval_precision"] for metric in metric_rows
            ),
            "retrieval_recall_mean_supported_queries": (
                statistics.mean(recalls) if recalls else None
            ),
            "retrieval_recall_query_count": len(recalls),
            "mean_noise_documents_before_filtering": statistics.mean(
                metric["noise_documents_before_filtering"] for metric in metric_rows
            ),
            "mean_noise_documents_after_filtering": statistics.mean(
                metric["noise_documents_after_filtering"] for metric in metric_rows
            ),
            "noise_reduction_overall": _safe_ratio(
                total_noise_before - total_noise_after, total_noise_before
            ),
            "expected_fact_coverage_mean_supported_queries": (
                statistics.mean(fact_coverage_values) if fact_coverage_values else None
            ),
            "expected_fact_coverage": _safe_ratio(
                expected_fact_hits, expected_fact_total
            ),
            "expected_fact_hits": expected_fact_hits,
            "expected_fact_total": expected_fact_total,
            "expected_fact_coverage_abstention_accuracy_mean_all_queries": statistics.mean(
                metric.get(
                    "expected_fact_coverage_abstention_accuracy",
                    metric["answer_accuracy"],
                )
                for metric in metric_rows
            ),
            "answer_accuracy_mean_all_queries": statistics.mean(
                metric["answer_accuracy"] for metric in metric_rows
            ),
            "expected_fact_coverage_abstention_accuracy_mean_all_queries": statistics.mean(
                metric["answer_accuracy"] for metric in metric_rows
            ),
            "correct_insufficient_evidence_count": correct_insufficient_evidence_count,
            "insufficient_evidence_query_count": insufficient_evidence_count,
            "correct_insufficient_evidence_rate": _safe_ratio(
                correct_insufficient_evidence_count, insufficient_evidence_count
            ),
            "supported_claim_count": supported_claims,
            "unsupported_claim_count": unsupported_claims,
            "contradicted_claim_count": contradicted_claims,
            "conflicted_claim_count": conflicted_claims,
            "unsupported_or_contradicted_claim_count": unsupported_or_contradicted_claims,
            "answer_claims_evaluated": total_claims,
            "supported_claim_rate": _safe_ratio(supported_claims, total_claims),
            "unsupported_claim_rate": _safe_ratio(unsupported_claims, total_claims),
            "contradicted_claim_rate": _safe_ratio(contradicted_claims, total_claims),
            "conflicted_claim_rate": _safe_ratio(conflicted_claims, total_claims),
            "unsupported_or_contradicted_claim_rate": _safe_ratio(
                unsupported_or_contradicted_claims, total_claims
            ),
            "unsupported_answer_claims": unsupported_or_contradicted_claims,
            "independent_hallucination_rate": _safe_ratio(
                unsupported_or_contradicted_claims, total_claims
            ),
            "pipeline_hallucination_rate_mean": statistics.mean(
                float(record["pipeline_hallucination_check"].get(
                    "hallucination_rate", 0.0
                ))
                for record in records
            ),
            "unreliable_documents_passed_to_generation_mean": statistics.mean(
                metric["unreliable_documents_passed_to_generation"]
                for metric in metric_rows
            ),
            "generation_context_characters_mean": statistics.mean(
                metric["generation_context_characters"] for metric in metric_rows
            ),
            "evidence_confidence_mean": statistics.mean(
                record["evidence_confidence_score"] or 0.0 for record in records
            ),
            "latency_seconds": _latency_statistics([
                record["measured_end_to_end_latency_seconds"] for record in records
            ]),
            "feedback_loop_trigger_count": sum(
                bool(record["feedback_loop"].get("triggered")) for record in records
            ),
            "feedback_loop_retry_count": sum(
                int(record["feedback_loop"].get("retry_count", 0))
                for record in records
            ),
            "feedback_loop_max_retries": max(
                (int(record["feedback_loop"].get("max_retries", 0)) for record in records),
                default=0,
            ),
        }
        supported_query_claims = [
            claim
            for row, record in zip(rows, records)
            if row["ground_truth"]["expected_behavior"] == "answer"
            for claim in record["metrics"]["independent_grounding_evaluation"]["claims"]
        ]
        aggregates[system_name]["supported_query_factual_claim_rate"] = _safe_ratio(
            sum(claim["status"] == "SUPPORTED" for claim in supported_query_claims),
            len(supported_query_claims),
        )
        aggregates[system_name]["supported_query_factual_claims"] = sum(
            claim["status"] == "SUPPORTED" for claim in supported_query_claims
        )
        aggregates[system_name]["supported_query_claims_evaluated"] = len(
            supported_query_claims
        )

    expected_conflicts = sum(
        row["conflict_evaluation"]["expected_conflict_count"] for row in rows
    )
    detected_conflicts = sum(
        row["conflict_evaluation"]["detected_expected_conflict_count"] for row in rows
    )
    aggregates["denoised"]["expected_conflict_detection_recall"] = _safe_ratio(
        detected_conflicts, expected_conflicts
    )
    aggregates["denoised"]["expected_conflicts_detected"] = detected_conflicts
    aggregates["denoised"]["expected_conflicts_total"] = expected_conflicts
    conflicting_query_count = sum(
        row["category"] == "conflicting" for row in rows
    )
    correctly_handled_conflict_queries = sum(
        row["category"] == "conflicting"
        and row["conflict_evaluation"]["expected_conflict_count"] > 0
        and row["conflict_evaluation"]["detected_expected_conflict_count"]
        == row["conflict_evaluation"]["expected_conflict_count"]
        for row in rows
    )
    aggregates["denoised"]["contradiction_handling_rate"] = _safe_ratio(
        correctly_handled_conflict_queries, conflicting_query_count
    )
    aggregates["denoised"]["correctly_handled_conflict_queries"] = (
        correctly_handled_conflict_queries
    )
    aggregates["denoised"]["conflicting_query_count"] = conflicting_query_count
    for system_name in ("standard", "denoised"):
        aggregates[system_name]["unsupported_query_abstention_rate"] = (
            aggregates[system_name]["correct_insufficient_evidence_rate"]
        )
        aggregates[system_name]["overall_unsupported_or_contradicted_claim_rate"] = (
            aggregates[system_name]["unsupported_or_contradicted_claim_rate"]
        )
        aggregates[system_name]["overall_unsupported_claim_rate"] = (
            aggregates[system_name]["unsupported_claim_rate"]
        )
    aggregates["by_category"] = {}
    for category in EXPECTED_CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]
        aggregates["by_category"][category] = {
            system_name: {
                "expected_fact_coverage_abstention_accuracy_mean": statistics.mean(
                    row["systems"][system_name]["metrics"].get(
                        "expected_fact_coverage_abstention_accuracy",
                        row["systems"][system_name]["metrics"]["answer_accuracy"],
                    )
                    for row in category_rows
                ),
                "expected_fact_coverage": (
                    _safe_ratio(
                        sum(len(row["systems"][system_name]["metrics"].get(
                            "matched_expected_facts", []
                        )) for row in category_rows),
                        sum(
                            len(row["ground_truth"]["expected_facts"])
                            for row in category_rows
                        ),
                    )
                    if any(row["ground_truth"]["expected_facts"] for row in category_rows)
                    else None
                ),
                "supported_claim_count": sum(
                    claim["status"] == "SUPPORTED"
                    for row in category_rows
                    for claim in row["systems"][system_name]["metrics"][
                        "independent_grounding_evaluation"
                    ]["claims"]
                ),
                "unsupported_claim_count": sum(
                    claim["status"] == "UNSUPPORTED"
                    for row in category_rows
                    for claim in row["systems"][system_name]["metrics"][
                        "independent_grounding_evaluation"
                    ]["claims"]
                ),
                "contradicted_claim_count": sum(
                    claim["status"] == "CONTRADICTED"
                    for row in category_rows
                    for claim in row["systems"][system_name]["metrics"][
                        "independent_grounding_evaluation"
                    ]["claims"]
                ),
                "conflicted_claim_count": sum(
                    claim["status"] == "CONFLICTED"
                    for row in category_rows
                    for claim in row["systems"][system_name]["metrics"][
                        "independent_grounding_evaluation"
                    ]["claims"]
                ),
                "factual_claim_count": sum(
                    row["systems"][system_name]["metrics"][
                        "independent_grounding_evaluation"
                    ]["claim_count"]
                    for row in category_rows
                ),
                "unsupported_query_abstention_rate": (
                    _safe_ratio(
                        sum(
                            row["systems"][system_name]["metrics"][
                                "correct_insufficient_evidence_behavior"
                            ] is True
                            for row in category_rows
                        ),
                        sum(
                            row["ground_truth"]["expected_behavior"]
                            == "insufficient_evidence"
                            for row in category_rows
                        ),
                    )
                    if any(
                        row["ground_truth"]["expected_behavior"]
                        == "insufficient_evidence"
                        for row in category_rows
                    )
                    else None
                ),
                "retrieval_precision_mean": statistics.mean(
                    row["systems"][system_name]["metrics"]["retrieval_precision"]
                    for row in category_rows
                ),
                "unsupported_claim_rate": _safe_ratio(
                    sum(
                        row["systems"][system_name]["metrics"][
                            "independent_grounding_evaluation"
                        ]["unsupported_claim_count"]
                        for row in category_rows
                    ),
                    sum(
                        row["systems"][system_name]["metrics"][
                            "independent_grounding_evaluation"
                        ]["claim_count"]
                        for row in category_rows
                    ),
                ),
                "unsupported_or_contradicted_claim_rate": _safe_ratio(
                    sum(
                        row["systems"][system_name]["metrics"][
                            "independent_grounding_evaluation"
                        ].get("unsupported_or_contradicted_claim_count", 0)
                        for row in category_rows
                    ),
                    sum(
                        row["systems"][system_name]["metrics"][
                            "independent_grounding_evaluation"
                        ]["claim_count"]
                        for row in category_rows
                    ),
                ),
            }
            for system_name in ("standard", "denoised")
        }
    return aggregates


def _format_metric(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def _write_report(payload: dict) -> str:
    aggregates = payload["aggregates"]
    lines = [
        "# Day 4 RAG Benchmark",
        "",
        f"- Queries: {payload['metadata']['query_count']} (5 per category)",
        "- Systems: Standard RAG and Denoised RAG, same corpus, query set, and top-k.",
        "- Generation: deterministic local extractive mode; external API keys are disabled.",
        f"- Semantic embeddings: `{payload['metadata']['embedding_model']}`",
        f"- NLI evaluator: `{payload['metadata']['nli_model']}`",
        f"- Pipeline/index and model initialization (outside query timers): "
        f"{payload['metadata']['initialization_seconds']:.3f} seconds",
        f"- NLI first-inference warm-up (outside query timers): "
        f"{payload['metadata']['nli_warmup_seconds']:.3f} seconds",
        "- Query latency: externally timed `pipeline.run` wall-clock duration; mean, "
        "median, nearest-rank p95, min, and max are reported in seconds.",
        "- Expected-fact coverage is normalized exact-token presence in the answer, "
        "not proof of factual correctness. The composite Expected-Fact Coverage / "
        "Abstention Accuracy averages answerable-query fact coverage with correct "
        "abstention on unsupported queries.",
        "- For conflict cases, expected facts are drawn from the higher-reliability "
        "corpus source; detecting/displaying the opposing claim is evaluated separately.",
        "- Retrieval precision/recall uses one shared initial top-k retrieval for each "
        "query; denoised feedback-retry outputs remain recorded separately.",
        "- Independent answer claims are compared with ground-truth relevant "
        f"corpus evidence using the NLI model at confidence >= "
        f"{CLAIM_CONFIDENCE_THRESHOLD:.2f}; only evidence meeting the existing "
        "source-trust threshold determines claim status.",
        "- Claims supported and contradicted by credible evidence are labeled "
        "CONFLICTED, not SUPPORTED. Supported, unsupported, contradicted, and "
        "conflicted claim rates use all extracted factual claims as their denominator.",
        "- Unsupported-claim rate excludes contradicted and conflicted claims. The "
        "separate unsupported-or-contradicted rate combines only unsupported and "
        "contradicted claims; conflicted claims remain separately reported.",
        "- The system's pipeline hallucination-detector findings and independent "
        "benchmark claim evaluation are separate measurements.",
        "- Zero-denominator policy: aggregate precision and rates are 0; category "
        "metrics with no applicable queries are n/a. Recall is excluded for unsupported "
        "queries and reported with its query count.",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | Standard RAG | Denoised RAG |",
        "|---|---:|---:|",
    ]
    metric_labels = (
        ("retrieval_precision_mean", "Retrieved-document precision"),
        ("retrieval_recall_mean_supported_queries", "Retrieved-document recall"),
        ("mean_noise_documents_before_filtering", "Mean irrelevant retrieved documents"),
        ("mean_noise_documents_after_filtering", "Mean irrelevant documents passed to generation"),
        ("noise_reduction_overall", "Noise reduction (overall)"),
        ("expected_fact_coverage", "Expected-fact coverage (fact-weighted token presence)"),
        (
            "expected_fact_coverage_abstention_accuracy_mean_all_queries",
            "Expected-fact coverage / abstention accuracy",
        ),
        ("correct_insufficient_evidence_rate", "Correct unsupported-query abstention rate"),
        ("pipeline_hallucination_rate_mean", "Pipeline detector hallucination rate"),
        ("unreliable_documents_passed_to_generation_mean", "Unreliable documents passed (mean)"),
        ("generation_context_characters_mean", "Generation context characters (mean)"),
        ("supported_query_factual_claim_rate", "Supported-query factual claim rate"),
        ("supported_claim_rate", "Supported claim rate (all factual claims)"),
        ("unsupported_claim_rate", "Unsupported claim rate (all factual claims)"),
        ("contradicted_claim_rate", "Contradicted claim rate (all factual claims)"),
        ("conflicted_claim_rate", "Conflicted/unresolved claim rate (all factual claims)"),
        ("unsupported_query_abstention_rate", "Unsupported-query abstention rate"),
        (
            "unsupported_or_contradicted_claim_rate",
            "Unsupported-or-contradicted claim rate (independent evaluator)",
        ),
        ("evidence_confidence_mean", "Evidence confidence score (mean; descriptive)"),
    )
    for key, label in metric_labels:
        standard_value = aggregates["standard"].get(key)
        denoised_value = aggregates["denoised"].get(key)
        suffix = "%" if key == "noise_reduction_overall" else ""
        if suffix:
            standard_value = standard_value * 100
            denoised_value = denoised_value * 100
        lines.append(
            f"| {label} | {_format_metric(standard_value, suffix)} | "
            f"{_format_metric(denoised_value, suffix)} |"
        )
    lines.extend([
        "",
        "| Latency statistic | Standard RAG (seconds) | Denoised RAG (seconds) |",
        "|---|---:|---:|",
    ])
    for statistic_name in ("mean", "median", "p95", "min", "max"):
        lines.append(
            f"| {statistic_name.title()} | "
            f"{aggregates['standard']['latency_seconds'][statistic_name]:.4f} | "
            f"{aggregates['denoised']['latency_seconds'][statistic_name]:.4f} |"
        )
    lines.extend([
        "",
        f"Denoised expected-conflict detection recall: "
        f"{_format_metric(aggregates['denoised']['expected_conflict_detection_recall'])} "
        f"({aggregates['denoised']['expected_conflicts_detected']}/"
        f"{aggregates['denoised']['expected_conflicts_total']}).",
        f"Denoised contradiction handling rate (correct conflicts / conflicting queries): "
        f"{_format_metric(aggregates['denoised']['contradiction_handling_rate'])} "
        f"({aggregates['denoised']['correctly_handled_conflict_queries']}/"
        f"{aggregates['denoised']['conflicting_query_count']}).",
        "",
        f"Denoised feedback retries: "
        f"{aggregates['denoised']['feedback_loop_retry_count']} across "
        f"{payload['metadata']['query_count']} queries "
        f"(maximum configured retries per query: "
        f"{aggregates['denoised']['feedback_loop_max_retries']}).",
        "",
        "## Results by category",
        "",
        "| Category | System | Expected-fact coverage | Supported | Unsupported | "
        "Contradicted | Conflicted | Claims generated | Unsupported abstention rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for category in EXPECTED_CATEGORIES:
        for system_name in ("standard", "denoised"):
            category_metrics = aggregates["by_category"][category][system_name]
            lines.append(
                f"| {category} | {system_name} | "
                f"{_format_metric(category_metrics['expected_fact_coverage'])} | "
                f"{category_metrics['supported_claim_count']} | "
                f"{category_metrics['unsupported_claim_count']} | "
                f"{category_metrics['contradicted_claim_count']} | "
                f"{category_metrics['conflicted_claim_count']} | "
                f"{category_metrics['factual_claim_count']} | "
                f"{_format_metric(category_metrics['unsupported_query_abstention_rate'])} |"
            )
    lines.extend([
        "",
        "## Per-query results",
        "",
        "| ID | Category | Standard expected-fact coverage / abstention accuracy | "
        "Denoised expected-fact coverage / abstention accuracy | "
        "Standard unsupported claims | Denoised unsupported claims | "
        "Standard latency (s) | Denoised latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["rows"]:
        standard = row["systems"]["standard"]
        denoised = row["systems"]["denoised"]
        lines.append(
            f"| {row['id']} | {row['category']} | "
            f"{standard['metrics']['expected_fact_coverage_abstention_accuracy']:.3f} | "
            f"{denoised['metrics']['expected_fact_coverage_abstention_accuracy']:.3f} | "
            f"{standard['metrics']['independent_grounding_evaluation']['unsupported_claim_count']} | "
            f"{denoised['metrics']['independent_grounding_evaluation']['unsupported_claim_count']} | "
            f"{standard['measured_end_to_end_latency_seconds']:.4f} | "
            f"{denoised['measured_end_to_end_latency_seconds']:.4f} |"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is a small, corpus-specific evaluation, not a broad generalization test.",
        "- Expected-fact scoring uses exact normalized token coverage against facts "
        "curated from the corpus; paraphrases may score lower.",
        "- Independent claim grounding depends on the selected NLI model and its "
        "confidence threshold; it is an evaluator, not a guarantee of factual truth.",
        "- Timing is warm execution time after local model initialization and a single "
        "NLI warm-up; it is not cold-start latency and is hardware-dependent.",
        "- The query dataset is curated and therefore should not be treated as a "
        "statistically representative sample.",
        "",
        "Full raw inputs, pipeline outputs, per-query evaluations, and aggregates are "
        "available in `per_query_results.json` and `summary.json`.",
        "",
    ])
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_benchmark() -> dict:
    documents, queries = load_inputs()
    initialization_started = time.perf_counter()
    # Force deterministic, offline extractive generation regardless of local API keys.
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "",
    }):
        standard, denoised = build_pipelines()
    initialization_seconds = time.perf_counter() - initialization_started
    index = standard.retrieval_agent.index
    nli_classifier = denoised.contradiction_agent.nli
    # The embedding index is built above; load/warm lazy NLI weights before timings.
    warmup_started = time.perf_counter()
    nli_classifier.classify_pairs([
        ("A vehicle is moving.", "A car is in motion."),
    ])
    nli_warmup_seconds = time.perf_counter() - warmup_started

    rows = [
        evaluate_query(query, standard, denoised, documents, nli_classifier)
        for query in queries
    ]
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_count": len(queries),
            "category_counts": dict(Counter(query["category"] for query in queries)),
            "corpus_document_count": len(documents),
            "top_k": TOP_K,
            "generation_mode": "deterministic_local_extractive",
            "embedding_model": index.model_name,
            "nli_model": nli_classifier.model_name,
            "max_retries": denoised.max_retries,
            "initialization_seconds": initialization_seconds,
            "nli_warmup_seconds": nli_warmup_seconds,
            "latency_method": "perf_counter around pipeline.run after model initialization and NLI warm-up",
            "query_dataset_sha256": _file_sha256(QUERY_PATH),
            "corpus_sha256": _file_sha256(DOCUMENT_PATH),
        },
        "aggregates": _aggregate(rows),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "per_query_results.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2))
    report = _write_report(payload)
    (RESULTS_DIR / "comparison.md").write_text(report)
    print(report)
    print(f"\nResults written under {RESULTS_DIR.relative_to(ROOT)}/")
    return payload


if __name__ == "__main__":
    run_benchmark()
