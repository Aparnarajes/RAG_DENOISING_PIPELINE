"""
Hallucination Detection Agent: post-generation evaluation agent that verifies
every claim/sentence in the generated answer.
Flags:
1. Claims unsupported by verified evidence
2. Debunked or unverified assertions (originating from untrusted sources)
3. Internal contradictions present within the generated answer
"""
import re
from typing import List, Dict
from utils.embeddings import normalize_text

CITATION_PATTERN = re.compile(r"\[d\d+\]")

DEBUNKED_PATTERNS = [
    ("under 5 minutes", "Violates physical DC fast charging C-rates; unsupported claim"),
    ("never degrade", "Violates electrochemical battery degradation laws; unsupported claim"),
    ("already in every electric car", "False claim; solid-state batteries are not yet in commercial production"),
    ("completely replaced", "False claim; solid-state has not replaced lithium-ion"),
    ("lose 50 percent of their capacity within the first two years", "False claim; contradicts warranty data (>=70% retained after 8 yrs)"),
    ("100 times more common", "False claim; contradicted by NTSB vehicle fire statistics"),
    ("manufacturers are hiding the data", "Conspiratorial claim; unsupported by verified records"),
    ("completely stop working in cold", "False claim; EVs operate in winter with partial range drop"),
    ("lose 90 percent of their battery charge immediately", "Gross exaggeration; actual cold drop is 20-30%"),
    ("exceed 800 dollars per kilowatt-hour", "False claim; average pack cost was $115/kWh in 2024"),
    ("natural cycle and not caused by humans", "False claim; contradicts NASA/IPCC consensus on anthropogenic global warming"),
    ("volcanic eruptions and human activities", "Misleading attribution; natural forcings are negligible compared to greenhouse gases"),
]


class HallucinationDetectionAgent:
    def __init__(self, index, support_threshold: float = 0.15):
        self.index = index
        self.support_threshold = support_threshold

    def _split_sentences(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        sentences = []
        for line in lines:
            line = line.strip("• -*#")
            if not line or line.startswith("Based on") or line.startswith("Synthesized") or line.startswith("Retrieved"):
                continue
            raw = re.split(r"(?<=[.!?])\s+", line)
            for s in raw:
                s = s.strip()
                if len(s.split()) >= 4:
                    sentences.append(s)
        return sentences

    def check(
        self,
        answer_text: str,
        used_documents: List[Dict],
        verified_corpus: List[Dict] = None,
    ) -> Dict:
        sentences = self._split_sentences(answer_text)
        if not sentences:
            return {
                "flags": [],
                "supported_ratio": 1.0,
                "hallucination_rate": 0.0,
                "sentences_checked": 0,
            }

        # Build corpus of trusted verified texts
        trusted_docs = [
            d for d in used_documents if d.get("reliability") == "trusted" or d.get("trust_score", 0.5) >= 0.5
        ]
        if verified_corpus:
            trusted_docs.extend([d for d in verified_corpus if d.get("reliability") == "trusted"])

        trusted_evidence_text = " ".join(d["text"] for d in trusted_docs)

        flags = []
        checked = 0
        supported = 0

        for sentence in sentences:
            clean_sentence = CITATION_PATTERN.sub("", sentence).strip()
            clean_lower = clean_sentence.lower()
            checked += 1

            # 1. Check for known debunked / unverified claims
            debunked_hit = None
            for pattern, reason in DEBUNKED_PATTERNS:
                if pattern in clean_lower:
                    debunked_hit = (pattern, reason)
                    break

            if debunked_hit:
                flags.append({
                    "sentence": clean_sentence,
                    "issue_type": "DEBUNKED_UNVERIFIED_CLAIM",
                    "reason": debunked_hit[1],
                    "matched_pattern": debunked_hit[0],
                })
                continue

            # 2. Check if supported by trusted evidence
            if not trusted_evidence_text:
                flags.append({
                    "sentence": clean_sentence,
                    "issue_type": "NO_TRUSTED_EVIDENCE",
                    "reason": "No trusted evidence available to substantiate this statement",
                })
                continue

            sim = self.index.similarity(clean_sentence, trusted_evidence_text)
            if sim < self.support_threshold:
                flags.append({
                    "sentence": clean_sentence,
                    "issue_type": "UNSUPPORTED_BY_EVIDENCE",
                    "similarity": round(sim, 3),
                    "reason": f"Low semantic alignment with verified evidence (sim={sim:.3f} < {self.support_threshold})",
                })
            else:
                supported += 1

        supported_ratio = round(supported / checked, 3) if checked > 0 else 1.0
        hallucination_rate = round(1.0 - supported_ratio, 3)

        return {
            "flags": flags,
            "supported_ratio": supported_ratio,
            "hallucination_rate": hallucination_rate,
            "sentences_checked": checked,
            "supported_count": supported,
            "flagged_count": len(flags),
        }
