"""
Denoised Multi-Agent RAG Pipeline:
Orchestrates the 5 specialized agents:
1. Retrieval Agent -> fetches raw candidates from index
2. Relevance Scoring Agent -> scores 0-1 & filters low-alignment docs
3. Evidence Verification Agent -> filters untrusted sources & red-flag claims
4. Contradiction Detection Agent -> detects & resolves factual conflicts
5. Answer Generation Agent -> synthesizes evidence-grounded output and evidence confidence
Followed by:
- Hallucination Detection Agent -> post-generation check
- Agent Feedback Loop -> bounded query expansion when evidence is insufficient
"""
import time
import os
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
        retry_evidence_confidence_threshold: float = 0.65,
        max_retries: int | None = None,
        retry_confidence_threshold: float | None = None,
    ):
        if retry_confidence_threshold is not None:
            retry_evidence_confidence_threshold = retry_confidence_threshold
        if not 0.0 <= retry_evidence_confidence_threshold <= 1.0:
            raise ValueError("Retry evidence-confidence threshold must be from 0 to 1.")
        if max_retries is None:
            configured_retries = os.getenv("RAG_MAX_RETRIES", "1")
            try:
                max_retries = int(configured_retries)
            except ValueError as exc:
                raise ValueError("RAG_MAX_RETRIES must be a non-negative integer.") from exc
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer.")
        self.retrieval_agent = retrieval_agent
        self.relevance_agent = relevance_agent
        self.verification_agent = verification_agent
        self.contradiction_agent = contradiction_agent
        self.answer_agent = answer_agent
        self.hallucination_agent = hallucination_agent
        self.retry_evidence_confidence_threshold = retry_evidence_confidence_threshold
        # Compatibility alias for callers using the previous constructor name.
        self.retry_confidence_threshold = retry_evidence_confidence_threshold
        self.max_retries = max_retries

    @staticmethod
    def _without_retrieval_scores(documents: List[Dict]) -> List[Dict]:
        return [
            {key: value for key, value in document.items() if key != "raw_score"}
            for document in documents
        ]

    def _generation_gate(self, query: str, documents: List[Dict]) -> List[Dict]:
        rescored = self.relevance_agent.score(
            query, self._without_retrieval_scores(documents)
        )
        relevant = self.relevance_agent.filter(rescored)
        return [
            document for document in relevant
            if (
                document.get("verification_status") == "VERIFIED"
                and float(document.get("trust_score", 0.0))
                >= self.verification_agent.trust_threshold
                and all(
                    claim.get("status") == "VERIFIED"
                    for claim in document.get("verification_claims", [])
                )
            )
        ]

    def _run_once(
        self,
        query: str,
        top_k: int,
        evidence_query: str | None = None,
    ) -> dict:
        evidence_query = evidence_query or query
        # Stage 1: Retrieval
        retrieved = self.retrieval_agent.retrieve(query, top_k=top_k)

        # Feedback retrieval can expand the search query, but evidence must
        # remain relevant to the original user question.
        scoring_documents = (
            self._without_retrieval_scores(retrieved)
            if evidence_query != query
            else retrieved
        )
        scored = self.relevance_agent.score(evidence_query, scoring_documents)
        relevance_filtered = self.relevance_agent.filter(scored)

        # Stage 3: Evidence Verification & Filtering
        verified = self.verification_agent.verify(relevance_filtered)

        # Analyze every candidate before filtering so conflicts with low-priority
        # sources remain visible in the conflict report.
        contradiction_analyzed, conflict_log = self.contradiction_agent.detect_and_resolve(
            verified
        )
        evidence_filtered = self.verification_agent.filter(contradiction_analyzed)
        generation_evidence = self._generation_gate(evidence_query, evidence_filtered)

        # Stage 5: Generate only when relevant, trusted, verified evidence survives.
        if generation_evidence:
            gen_result = self.answer_agent.generate(
                evidence_query,
                generation_evidence,
                is_standard_rag=False,
                conflicts=conflict_log,
            )
            abstention_reason = None
        else:
            abstention_reason = "No sufficiently relevant verified evidence"
            gen_result = {
                "answer": (
                    "Insufficient verified evidence in the retrieved knowledge base "
                    "to answer this question."
                ),
                "citations": [],
                "structured_citations": [],
                "confidence": 0.0,
                "evidence_confidence": 0.0,
                "evidence_confidence_score": 0.0,
                "abstained": True,
                "abstention_reason": abstention_reason,
                "evidence": [],
            }

        # An abstention is not a factual answer claim and should not be flagged
        # as unsupported by the post-generation checker.
        if gen_result.get("abstained", False):
            hallucination_check = {
                "claims": [],
                "flags": [],
                "supported_ratio": 1.0,
                "hallucination_rate": 0.0,
                "sentences_checked": 0,
                "supported_count": 0,
                "flagged_count": 0,
            }
        else:
            hallucination_check = self.hallucination_agent.check(
                gen_result["answer"], generation_evidence
            )

        # Format explicit relevance scores matching Section 3.3
        relevance_scores = [
            {
                "document_id": d["document_id"],
                "score": d["score"],
                "reasoning": d["reasoning"],
                "decision": d["decision"],
                # Keep compatibility with benchmark/dashboard consumers.
                "doc_id": d["document_id"],
                "relevance_score": d["score"],
            }
            for d in scored
        ]

        return {
            "query": query,
            "pipeline_type": "denoised_rag",
            "retrieved_documents": retrieved,
            "relevance_scored_documents": scored,
            "evidence_verified_documents": verified,
            "filtered_documents": generation_evidence,
            "generation_evidence": [
                document.get("id", document.get("document_id"))
                for document in generation_evidence
            ],
            "abstained": bool(gen_result.get("abstained", False)),
            "abstention_reason": gen_result.get("abstention_reason", abstention_reason),
            "relevance_scores": relevance_scores,
            "conflicts_detected": conflict_log,
            "final_answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "structured_citations": gen_result["structured_citations"],
            "evidence_confidence_score": gen_result.get(
                "evidence_confidence_score",
                gen_result.get("evidence_confidence", gen_result["confidence"]),
            ),
            "confidence_score": gen_result.get(
                "evidence_confidence_score",
                gen_result.get("evidence_confidence", gen_result["confidence"]),
            ),
            "confidence": gen_result.get(
                "evidence_confidence_score",
                gen_result.get("evidence_confidence", gen_result["confidence"]),
            ),
            "hallucination_check": hallucination_check,
        }

    def run(self, query: str, top_k: int = 6) -> dict:
        start = time.time()
        result = self._run_once(query, top_k=top_k, evidence_query=query)
        best_result = result
        attempts = []
        expanded_query = query
        retry_count = 0
        current_result = result
        improved = False

        while retry_count < self.max_retries:
            insufficient_evidence = (
                current_result["abstained"]
                or current_result["evidence_confidence_score"]
                < self.retry_evidence_confidence_threshold
                or len(current_result["generation_evidence"]) < 2
            )
            if not insufficient_evidence or not current_result["retrieved_documents"]:
                break

            candidate_text = " ".join(
                str(document.get("text", ""))
                for document in current_result["retrieved_documents"][:3]
            )
            new_terms = self.retrieval_agent.index.top_terms_for_text(
                candidate_text, n=4
            )
            existing_terms = set(expanded_query.lower().split())
            new_terms = [
                term for term in new_terms
                if term.lower() not in existing_terms
            ]
            if not new_terms:
                break

            expanded_query = f"{expanded_query} {' '.join(new_terms)}".strip()
            retry_count += 1
            retry_result = self._run_once(
                expanded_query, top_k=top_k, evidence_query=query
            )
            retry_result["query"] = query
            attempts.append({
                "retry": retry_count,
                "expanded_query": expanded_query,
                "evidence_confidence_score": retry_result["evidence_confidence_score"],
                "abstained": retry_result["abstained"],
                "generation_evidence": retry_result["generation_evidence"],
            })

            if (
                retry_result["evidence_confidence_score"]
                > best_result["evidence_confidence_score"]
                or (
                    best_result["abstained"]
                    and not retry_result["abstained"]
                )
            ):
                best_result = retry_result
                improved = True
            current_result = retry_result

        result = best_result
        result["feedback_loop"] = {
            "triggered": retry_count > 0,
            "original_query": query,
            "max_retries": self.max_retries,
            "retry_count": retry_count,
            "attempts": attempts,
            "improved": improved,
        }
        result["latency_seconds"] = round(time.time() - start, 4)
        return result
