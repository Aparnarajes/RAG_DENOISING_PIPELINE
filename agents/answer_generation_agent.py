"""
Answer Generation Agent: synthesizes the final answer using only the
evidence documents supplied to it, attaches citations linking claims to
source documents, and computes a descriptive evidence-strength score.
"""
import os
from typing import List, Dict


class AnswerGenerationAgent:
    def __init__(self):
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def generate(
        self,
        query: str,
        documents: List[Dict],
        is_standard_rag: bool = False,
        conflicts: List[Dict] | None = None,
    ) -> Dict:
        if not documents:
            return {
                "answer": "Insufficient verified evidence is available to answer this question reliably.",
                "citations": [],
                "structured_citations": [],
                "evidence_confidence_score": 0.0,
                "evidence_confidence": 0.0,
                "confidence": 0.0,
                "abstained": True,
                "abstention_reason": "No sufficiently relevant verified evidence",
                "evidence": [],
            }

        conflicts = conflicts or []
        if self.anthropic_key:
            answer_text = self._generate_with_anthropic(query, documents, conflicts)
        elif self.gemini_key:
            answer_text = self._generate_with_gemini(query, documents, conflicts)
        else:
            answer_text = self._generate_synthesized(
                query, documents, is_standard_rag, conflicts
            )

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

        # This score describes evidence quality; it is not model certainty or
        # a probability that the generated answer is correct.
        relevances = [d.get("relevance_score", d.get("raw_score", 0.3)) for d in documents]
        trusts = [d.get("trust_score", 0.2 if d.get("reliability") == "unreliable" else 0.8) for d in documents]

        avg_rel = sum(relevances) / len(relevances)
        avg_trust = sum(trusts) / len(trusts)

        # Penalty for untrusted documents in the generation context (e.g. in Standard RAG)
        untrusted_penalty = sum(0.15 for d in documents if d.get("reliability") == "unreliable")

        evidence_confidence = round(
            0.45 * avg_rel + 0.55 * avg_trust - untrusted_penalty, 3
        )
        evidence_confidence = max(0.0, min(1.0, evidence_confidence))

        return {
            "answer": answer_text,
            "citations": cited_doc_ids,
            "structured_citations": structured_citations,
            "evidence_confidence_score": evidence_confidence,
            "evidence_confidence": evidence_confidence,
            # Compatibility alias; consumers should prefer evidence_confidence.
            "confidence": evidence_confidence,
            "abstained": False,
            "abstention_reason": None,
            "evidence": cited_doc_ids,
        }

    @staticmethod
    def _format_conflicts(conflicts: List[Dict]) -> str:
        lines = [
            "Conflicting claims were detected; the claims below are reported as "
            "disputed, not as settled facts:"
        ]
        for conflict in conflicts:
            if conflict.get("relationship") != "CONTRADICTION":
                continue
            priority_a = conflict.get("source_priority_a")
            priority_b = conflict.get("source_priority_b")
            source_a = str(conflict.get("document_a", conflict.get("doc_a", "unknown")))
            source_b = str(conflict.get("document_b", conflict.get("doc_b", "unknown")))
            priority_text_a = (
                f", source priority {float(priority_a):.2f}"
                if priority_a is not None else ""
            )
            priority_text_b = (
                f", source priority {float(priority_b):.2f}"
                if priority_b is not None else ""
            )
            lines.append(
                f"- [{source_a}{priority_text_a}] {conflict['claim_a']} "
                f"vs. [{source_b}{priority_text_b}] {conflict['claim_b']} "
                f"(confidence {float(conflict.get('confidence', 0.0)):.2f})."
            )
            lines.append(f"  Resolution: {conflict.get('resolution', 'Unresolved.')}")
        return "\n".join(lines)

    def _generate_synthesized(
        self,
        query: str,
        documents: List[Dict],
        is_standard_rag: bool,
        conflicts: List[Dict] | None = None,
    ) -> str:
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
            if conflicts:
                lines.append(self._format_conflicts(conflicts))
            for doc in documents:
                lines.append(f"• {doc['text']} [{doc['id']}]")

        return "\n".join(lines)

    def _generate_with_anthropic(
        self, query: str, documents: List[Dict], conflicts: List[Dict] | None = None
    ) -> str:
        import urllib.request
        import json as _json

        context = "\n".join(f"[{d['id']}] (Source: {d.get('source')}) {d['text']}" for d in documents)
        prompt = (
            "You are an accurate, objective assistant. Answer the user question using ONLY "
            "the provided evidence. Attach inline citations like [d1] to every claim. "
            "If evidence contains contradictions, describe both claims as disputed and "
            "explain any supplied source-priority resolution. Never present a disputed "
            "claim as settled.\n\n"
            f"Evidence:\n{context}\n\n"
            f"Detected conflicts:\n{self._format_conflicts(conflicts or [])}\n\n"
            f"Question: {query}"
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

    def _generate_with_gemini(
        self, query: str, documents: List[Dict], conflicts: List[Dict] | None = None
    ) -> str:
        import urllib.request
        import json as _json

        context = "\n".join(f"[{d['id']}] {d['text']}" for d in documents)
        prompt = (
            f"Question: {query}\n\n"
            f"Evidence:\n{context}\n\n"
            "Synthesize a factual answer citing document IDs like [d1] for every factual statement. "
            "Describe detected conflicts as disputed claims, not settled facts, and explain "
            "any supplied source-priority resolution.\n\n"
            f"Detected conflicts:\n{self._format_conflicts(conflicts or [])}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        body = _json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
        return self._generate_synthesized(query, documents, False, conflicts)
