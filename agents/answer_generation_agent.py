"""
Answer Generation Agent: synthesizes the final answer using only the
evidence documents supplied to it, attaches citations linking claims to
source documents, and computes a model-estimated confidence score.
"""
import os
import re
from typing import List, Dict


class AnswerGenerationAgent:
    def __init__(self):
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def generate(self, query: str, documents: List[Dict], is_standard_rag: bool = False) -> Dict:
        if not documents:
            return {
                "answer": "Insufficient verified evidence is available to answer this question reliably.",
                "citations": [],
                "structured_citations": [],
                "confidence": 0.0,
            }

        if self.anthropic_key:
            answer_text = self._generate_with_anthropic(query, documents)
        elif self.gemini_key:
            answer_text = self._generate_with_gemini(query, documents)
        else:
            answer_text = self._generate_synthesized(query, documents, is_standard_rag)

        # Build citation list and granular claim-to-document mapping
        cited_doc_ids = [d["id"] for d in documents]
        structured_citations = []
        for d in documents:
            structured_citations.append({
                "doc_id": d["id"],
                "source": d.get("source", "Unknown"),
                "reliability": d.get("reliability", "unknown"),
                "trust_score": d.get("trust_score", 0.5),
                "relevance_score": d.get("relevance_score", d.get("raw_score", 0.5)),
                "claim_snippet": d["text"][:120] + "...",
            })

        # Confidence estimation:
        # Blends evidence relevance, source trustworthiness, and consensus
        relevances = [d.get("relevance_score", d.get("raw_score", 0.3)) for d in documents]
        trusts = [d.get("trust_score", 0.2 if d.get("reliability") == "unreliable" else 0.8) for d in documents]

        avg_rel = sum(relevances) / len(relevances)
        avg_trust = sum(trusts) / len(trusts)

        # Penalty for untrusted documents in the generation context (e.g. in Standard RAG)
        untrusted_penalty = sum(0.15 for d in documents if d.get("reliability") == "unreliable")

        confidence = round(0.45 * avg_rel + 0.55 * avg_trust - untrusted_penalty, 3)
        confidence = max(0.05, min(0.98, confidence))

        return {
            "answer": answer_text,
            "citations": cited_doc_ids,
            "structured_citations": structured_citations,
            "confidence": confidence,
        }

    def _generate_synthesized(self, query: str, documents: List[Dict], is_standard_rag: bool) -> str:
        """Generates a cohesive, cited answer from the provided documents.
        If run under standard RAG (no denoising), conflicting/unreliable docs
        are included, highlighting the contrast with the clean denoised output."""
        lines = []

        if is_standard_rag:
            lines.append(f"Retrieved information for \"{query}\":")
            for doc in documents:
                prefix = "• "
                if doc.get("reliability") == "unreliable":
                    prefix = "• [UNVERIFIED / CONFLICTING] "
                lines.append(f"{prefix}{doc['text']} [{doc['id']}]")
        else:
            lines.append(f"Synthesized answer based on verified evidence for \"{query}\":")
            for doc in documents:
                lines.append(f"• {doc['text']} [{doc['id']}]")

        return "\n".join(lines)

    def _generate_with_anthropic(self, query: str, documents: List[Dict]) -> str:
        import urllib.request
        import json as _json

        context = "\n".join(f"[{d['id']}] (Source: {d.get('source')}) {d['text']}" for d in documents)
        prompt = (
            "You are an accurate, objective assistant. Answer the user question using ONLY "
            "the provided evidence. Attach inline citations like [d1] to every claim. "
            "If evidence contains contradictions or unverified rumors, point them out explicitly.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {query}"
        )
        body = _json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 450,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return "".join(block.get("text", "") for block in data.get("content", []))

    def _generate_with_gemini(self, query: str, documents: List[Dict]) -> str:
        import urllib.request
        import json as _json

        context = "\n".join(f"[{d['id']}] {d['text']}" for d in documents)
        prompt = (
            f"Question: {query}\n\n"
            f"Evidence:\n{context}\n\n"
            "Synthesize a factual answer citing document IDs like [d1] for every factual statement."
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        body = _json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
        return self._generate_synthesized(query, documents, False)
