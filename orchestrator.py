"""
Orchestrator: runs user queries through both the standard RAG baseline
and the denoised multi-agent RAG pipeline, displaying a side-by-side
comparison of all outputs specified in Section 3.3.

Usage:
    python orchestrator.py "How long does it take to fast charge an EV battery?"
    python orchestrator.py            # runs all benchmark queries
"""
import sys
import json
from pathlib import Path

from utils.embeddings import TfidfIndex
from agents.retrieval_agent import RetrievalAgent
from agents.relevance_scoring_agent import RelevanceScoringAgent
from agents.evidence_verification_agent import EvidenceVerificationAgent
from agents.contradiction_detection_agent import ContradictionDetectionAgent
from agents.answer_generation_agent import AnswerGenerationAgent
from agents.hallucination_detection_agent import HallucinationDetectionAgent
from pipeline.standard_rag import StandardRAGPipeline
from pipeline.denoised_rag import DenoisedRAGPipeline

DATA_DIR = Path(__file__).parent / "data"


def build_pipelines():
    documents = json.loads((DATA_DIR / "documents.json").read_text())
    index = TfidfIndex(documents)

    retrieval_agent = RetrievalAgent(index)
    relevance_agent = RelevanceScoringAgent(index, threshold=0.20)
    verification_agent = EvidenceVerificationAgent(trust_threshold=0.5)
    contradiction_agent = ContradictionDetectionAgent(numeric_ratio_threshold=1.8)
    answer_agent = AnswerGenerationAgent()
    hallucination_agent = HallucinationDetectionAgent(index, support_threshold=0.15)

    standard = StandardRAGPipeline(retrieval_agent, answer_agent, hallucination_agent)
    denoised = DenoisedRAGPipeline(
        retrieval_agent=retrieval_agent,
        relevance_agent=relevance_agent,
        verification_agent=verification_agent,
        contradiction_agent=contradiction_agent,
        answer_agent=answer_agent,
        hallucination_agent=hallucination_agent,
        retry_confidence_threshold=0.65,
    )
    return standard, denoised


def print_comparison(query: str, standard_res: dict, denoised_res: dict):
    print("=" * 95)
    print(f"QUERY: {query}")
    print("=" * 95)

    print("\n[1] STANDARD RAG (Baseline)")
    print(f"  • Retrieved: {len(standard_res['retrieved_documents'])} docs -> Used all {len(standard_res['filtered_documents'])} for generation (No filtering)")
    for d in standard_res["retrieved_documents"]:
        rel_tag = d.get('reliability', 'unknown')
        print(f"    [{d['id']}] ({rel_tag:>10}) sim={d.get('raw_score', 0):.3f} | {d['text'][:70]}...")

    hc_std = standard_res["hallucination_check"]
    print(f"  • Confidence: {standard_res['confidence_score']:.3f} | Latency: {standard_res['latency_seconds']}s")
    print(f"  • Hallucination Supported Ratio: {hc_std['supported_ratio']:.2f} ({len(hc_std['flags'])} flagged claim(s))")
    for f in hc_std["flags"]:
        print(f"    ⚠️  FLAGGED: \"{f['sentence'][:70]}...\" -> {f.get('reason')}")
    print("  • Answer (Standard RAG):")
    for line in standard_res["final_answer"].split("\n"):
        print(f"    {line}")

    print("\n[2] DENOISED MULTI-AGENT RAG")
    print(f"  • Step 1 (Retrieval): {len(denoised_res['retrieved_documents'])} raw candidates fetched")
    print(f"  • Step 2 (Relevance Scoring): {len(denoised_res['relevance_scored_documents'])} scored -> {len([d for d in denoised_res['relevance_scored_documents'] if d.get('relevance_score',0)>=0.14])} passed threshold")
    for s in denoised_res["relevance_scores"][:3]:
        print(f"      doc {s['doc_id']}: score={s['relevance_score']} ({s['reasoning']})")

    print(f"  • Step 3 (Evidence Verification): {len(denoised_res['evidence_verified_documents'])} checked for source trust & misinformation")
    print(f"  • Step 4 (Contradiction Resolution): {len(denoised_res['filtered_documents'])} surviving clean evidence documents")
    if denoised_res["conflicts_detected"]:
        for c in denoised_res["conflicts_detected"]:
            print(f"      ⚡ Conflict on '{c['facet']}': {c['doc_a']} ({c['claim_a']}) vs {c['doc_b']} ({c['claim_b']})")
            print(f"         Resolution: {c['resolution']}")

    fb = denoised_res["feedback_loop"]
    if fb["triggered"]:
        print(f"  • Step 6 (Feedback Loop): TRIGGERED ({fb.get('reason')})")
        print(f"      Expanded Query: \"{fb['expanded_query']}\"")
        print(f"      Confidence Lift: {fb.get('original_confidence')} -> {fb.get('retry_confidence')} (improved={fb.get('improved')})")
    else:
        print(f"  • Step 6 (Feedback Loop): Not needed (high confidence: {denoised_res['confidence_score']:.3f})")

    hc_den = denoised_res["hallucination_check"]
    print(f"  • Confidence: {denoised_res['confidence_score']:.3f} | Latency: {denoised_res['latency_seconds']}s")
    print(f"  • Hallucination Supported Ratio: {hc_den['supported_ratio']:.2f} ({len(hc_den['flags'])} flagged)")
    print("  • Clean Final Answer (with citations):")
    for line in denoised_res["final_answer"].split("\n"):
        print(f"    {line}")
    print()


def main():
    standard, denoised = build_pipelines()

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [q["query"] for q in json.loads((DATA_DIR / "queries.json").read_text())]

    for q in queries:
        std_res = standard.run(q)
        den_res = denoised.run(q)
        print_comparison(q, std_res, den_res)


if __name__ == "__main__":
    main()
