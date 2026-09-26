"""
Standard RAG Baseline Pipeline:
Directly passes all retrieved candidate documents into the generation step
without relevance filtering, evidence verification, or contradiction detection.
Post-generation hallucination detection is run to enable empirical comparison.
"""
import time
from agents.retrieval_agent import RetrievalAgent
from agents.answer_generation_agent import AnswerGenerationAgent
from agents.hallucination_detection_agent import HallucinationDetectionAgent


class StandardRAGPipeline:
    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        answer_agent: AnswerGenerationAgent,
        hallucination_agent: HallucinationDetectionAgent,
    ):
        self.retrieval_agent = retrieval_agent
        self.answer_agent = answer_agent
        self.hallucination_agent = hallucination_agent

    def run(self, query: str, top_k: int = 6) -> dict:
        start = time.time()
        retrieved = self.retrieval_agent.retrieve(query, top_k=top_k)

        # Baseline skips all intermediate filtering and passes raw candidates to LLM
        gen_result = self.answer_agent.generate(query, retrieved, is_standard_rag=True)

        # Evaluate hallucination on the generated answer
        hallucination_check = self.hallucination_agent.check(gen_result["answer"], retrieved)

        latency = round(time.time() - start, 4)

        # Raw scores for the retrieved documents
        relevance_scores = [
            {
                "doc_id": d["id"],
                "raw_score": d.get("raw_score", 0.0),
                "relevance_score": d.get("raw_score", 0.0),
                "reasoning": f"Unfiltered retrieval raw similarity = {d.get('raw_score', 0.0):.3f}",
            }
            for d in retrieved
        ]

        return {
            "query": query,
            "pipeline_type": "standard_rag",
            "retrieved_documents": retrieved,
            "filtered_documents": retrieved,  # Baseline performs no filtering
            "relevance_scores": relevance_scores,
            "final_answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "structured_citations": gen_result["structured_citations"],
            "confidence_score": gen_result["confidence"],
            "confidence": gen_result["confidence"],
            "conflicts_detected": [],
            "hallucination_check": hallucination_check,
            "feedback_loop": {"triggered": False},
            "latency_seconds": latency,
        }
