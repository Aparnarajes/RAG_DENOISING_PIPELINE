"""
Denoised Multi-Agent RAG Pipeline:
Orchestrates the 5 specialized agents:
1. Retrieval Agent -> fetches raw candidates from index
2. Relevance Scoring Agent -> scores 0-1 & filters low-alignment docs
3. Evidence Verification Agent -> filters untrusted sources & red-flag claims
4. Contradiction Detection Agent -> detects & resolves factual conflicts
5. Answer Generation Agent -> synthesizes clean answer with citations + confidence
Followed by:
- Hallucination Detection Agent -> post-generation check
- Agent Feedback Loop -> expands query and retries if confidence is low
"""
import time
from typing import List, Dict
from agents.retrieval_agent import RetrievalAgent
from agents.relevance_scoring_agent import RelevanceScoringAgent
from agents.evidence_verification_agent import EvidenceVerificationAgent
from agents.contradiction_detection_agent import ContradictionDetectionAgent
from agents.answer_generation_agent import AnswerGenerationAgent
from agents.hallucination_detection_agent import HallucinationDetectionAgent


class DenoisedRAGPipeline:
    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        relevance_agent: RelevanceScoringAgent,
        verification_agent: EvidenceVerificationAgent,
        contradiction_agent: ContradictionDetectionAgent,
        answer_agent: AnswerGenerationAgent,
        hallucination_agent: HallucinationDetectionAgent,
        retry_confidence_threshold: float = 0.65,
    ):
        self.retrieval_agent = retrieval_agent
        self.relevance_agent = relevance_agent
        self.verification_agent = verification_agent
        self.contradiction_agent = contradiction_agent
        self.answer_agent = answer_agent
        self.hallucination_agent = hallucination_agent
        self.retry_confidence_threshold = retry_confidence_threshold

    def _run_once(self, query: str, top_k: int) -> dict:
        # Stage 1: Retrieval
        retrieved = self.retrieval_agent.retrieve(query, top_k=top_k)

        # Stage 2: Relevance Scoring & Filtering
        scored = self.relevance_agent.score(query, retrieved)
        relevance_filtered = self.relevance_agent.filter(scored)

        # Stage 3: Evidence Verification & Filtering
        verified = self.verification_agent.verify(relevance_filtered)
        evidence_filtered = self.verification_agent.filter(verified)

        # Stage 4: Contradiction Detection & Resolution
        surviving_docs, conflict_log = self.contradiction_agent.detect_and_resolve(
            evidence_filtered
        )

        # Stage 5: Answer Generation from clean document set only
        gen_result = self.answer_agent.generate(query, surviving_docs, is_standard_rag=False)

        # Bonus: Post-generation Hallucination Detection
        hallucination_check = self.hallucination_agent.check(
            gen_result["answer"], surviving_docs
        )

        # Format explicit relevance scores matching Section 3.3
        relevance_scores = [
            {
                "doc_id": d["id"],
                "relevance_score": d["relevance_score"],
                "reasoning": d["relevance_reasoning"],
            }
            for d in scored
        ]

        return {
            "query": query,
            "pipeline_type": "denoised_rag",
            "retrieved_documents": retrieved,
            "relevance_scored_documents": scored,
            "evidence_verified_documents": verified,
            "filtered_documents": surviving_docs,
            "relevance_scores": relevance_scores,
            "conflicts_detected": conflict_log,
            "final_answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "structured_citations": gen_result["structured_citations"],
            "confidence_score": gen_result["confidence"],
            "confidence": gen_result["confidence"],
            "hallucination_check": hallucination_check,
        }

    def run(self, query: str, top_k: int = 6) -> dict:
        start = time.time()
        result = self._run_once(query, top_k=top_k)

        feedback_loop = {"triggered": False}
        low_confidence = result["confidence"] < self.retry_confidence_threshold
        few_docs = len(result["filtered_documents"]) < 2

        # Trigger feedback loop if confidence is low or evidence was over-filtered
        if (low_confidence or few_docs) and result["retrieved_documents"]:
            # Extract key terms from the initial retrieved candidates
            candidate_text = " ".join(d["text"] for d in result["retrieved_documents"][:3])
            new_terms = self.retrieval_agent.index.top_terms_for_text(candidate_text, n=4)
            new_terms = [t for t in new_terms if t.lower() not in query.lower()]

            if new_terms:
                expanded_query = f"{query} {' '.join(new_terms)}".strip()
                feedback_loop = {
                    "triggered": True,
                    "reason": "Low initial confidence or sparse surviving evidence",
                    "original_confidence": result["confidence"],
                    "expanded_query": expanded_query,
                }

                retry_result = self._run_once(expanded_query, top_k=top_k)
                feedback_loop["retry_confidence"] = retry_result["confidence"]
                feedback_loop["improved"] = retry_result["confidence"] > result["confidence"]

                if retry_result["confidence"] > result["confidence"]:
                    retry_result["query"] = query  # preserve original user query
                    retry_result["relevance_scores"] = result["relevance_scores"]  # keep original scores
                    result = retry_result

        result["feedback_loop"] = feedback_loop
        result["latency_seconds"] = round(time.time() - start, 4)
        return result
