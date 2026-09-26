"""Shared claim extraction, NLI comparison, and configurable source policy."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Protocol, Sequence

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
MIN_NLI_SEMANTIC_SIMILARITY = 0.2
DEFAULT_SOURCE_PRIORITIES = {
    "official_government": 1.0,
    "academic_research": 0.9,
    "established_publication": 0.8,
    "organization_company": 0.7,
    "general_website": 0.55,
    "user_generated_forum": 0.2,
}
MIN_VERIFIED_SOURCE_SCORE = 0.5


class PairClassifier(Protocol):
    def classify_pairs(self, pairs: Sequence[tuple[str, str]]) -> List[Dict]: ...


def extract_claims(text: str, min_words: int = 2) -> List[str]:
    """Use complete sentences as auditable claim candidates."""
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    claims = []
    for candidate in candidates:
        claim = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", candidate).strip()
        claim = re.sub(r"\s*\[[^\]]+\]\s*$", "", claim).strip()
        if claim.endswith(":"):
            continue
        if len(claim.split()) >= min_words:
            claims.append(claim)
    return claims


def _normalise_category(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_")


class SourceReliabilityPolicy:
    """Maps explicit source categories or legacy reliability labels to scores."""

    def __init__(
        self,
        priorities: Mapping[str, float] | None = None,
        min_trusted_score: float = MIN_VERIFIED_SOURCE_SCORE,
    ):
        configured = priorities
        if configured is None:
            raw_priorities = os.getenv("RAG_SOURCE_PRIORITIES")
            if raw_priorities:
                try:
                    configured = json.loads(raw_priorities)
                except json.JSONDecodeError as exc:
                    raise ValueError("RAG_SOURCE_PRIORITIES must contain a JSON object.") from exc

        self.priorities = dict(DEFAULT_SOURCE_PRIORITIES)
        if configured is not None:
            if not isinstance(configured, Mapping):
                raise ValueError("Source priorities must be a mapping of categories to scores.")
            for category, score in configured.items():
                if not isinstance(category, str) or not isinstance(score, (int, float)):
                    raise ValueError("Each source priority must have a string category and numeric score.")
                if not 0.0 <= float(score) <= 1.0:
                    raise ValueError("Source priority scores must be between 0 and 1.")
                self.priorities[_normalise_category(category)] = float(score)

        if not 0.0 <= min_trusted_score <= 1.0:
            raise ValueError("Minimum trusted source score must be between 0 and 1.")
        self.min_trusted_score = min_trusted_score

    def score(self, document: Dict) -> float:
        category = document.get("source_category", document.get("source_type"))
        if category:
            normalized_category = _normalise_category(str(category))
            if normalized_category in self.priorities:
                return self.priorities[normalized_category]

        reliability = str(document.get("reliability", "")).strip().lower()
        if reliability == "trusted":
            return 0.75
        if reliability in {"unreliable", "untrusted"}:
            return 0.1
        return 0.0

    def is_trusted(self, document: Dict) -> bool:
        return self.score(document) >= self.min_trusted_score


@dataclass(frozen=True)
class _Relation:
    label: str
    confidence: float


class NLIComparator:
    """Lazy, local natural-language-inference model for claim-pair comparison."""

    def __init__(self, model_name: str | None = None, model=None, semantic_index=None):
        self.model_name = model_name or os.getenv("RAG_NLI_MODEL", DEFAULT_NLI_MODEL)
        self.model = model
        self.semantic_index = semantic_index

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is required for NLI comparison. "
                    "Install project dependencies with `pip install -r requirements.txt`."
                ) from exc
            self.model = CrossEncoder(self.model_name, max_length=512)
        return self.model

    def classify_pairs(self, pairs: Sequence[tuple[str, str]]) -> List[Dict]:
        if not pairs:
            return []
        model = self._get_model()
        scores = model.predict(
            list(pairs),
            show_progress_bar=False,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        semantic_index = self._get_semantic_index()
        semantic_scores = semantic_index.pairwise_similarities(pairs)
        label_map = {
            int(index): str(label).lower()
            for index, label in model.model.config.id2label.items()
        }
        results = []
        for (premise, hypothesis), row, semantic_score in zip(pairs, scores, semantic_scores):
            probabilities = [float(value) for value in row]
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            label = label_map.get(best_index, "").lower()
            if label in {"label_0", "label_1", "label_2"}:
                label = {
                    0: "contradiction",
                    1: "entailment",
                    2: "neutral",
                }[best_index]
            if label not in {"entailment", "contradiction", "neutral"}:
                raise ValueError(f"Unsupported NLI model label: {label or best_index}")
            if (
                label != "neutral"
                and semantic_score < MIN_NLI_SEMANTIC_SIMILARITY
            ):
                label = "neutral"
                neutral_index = next(
                    (index for index, mapped in label_map.items() if mapped == "neutral"),
                    None,
                )
                if neutral_index is not None:
                    confidence = probabilities[neutral_index]
                else:
                    confidence = 0.0
            else:
                confidence = probabilities[best_index]
            results.append({
                "relationship": label.upper(),
                "confidence": confidence,
                "semantic_similarity": semantic_score,
                "probabilities": {
                    label_map.get(index, str(index)).upper(): probability
                    for index, probability in enumerate(probabilities)
                },
            })
        return results

    def _get_semantic_index(self):
        if self.semantic_index is None:
            from utils.embeddings import SentenceTransformerIndex

            self.semantic_index = SentenceTransformerIndex([])
        return self.semantic_index


_NLI_COMPARATORS: Dict[str, NLIComparator] = {}


def get_nli_comparator(model_name: str | None = None, semantic_index=None) -> NLIComparator:
    name = model_name or os.getenv("RAG_NLI_MODEL", DEFAULT_NLI_MODEL)
    if name not in _NLI_COMPARATORS:
        _NLI_COMPARATORS[name] = NLIComparator(name, semantic_index=semantic_index)
    elif semantic_index is not None:
        _NLI_COMPARATORS[name].semantic_index = semantic_index
    return _NLI_COMPARATORS[name]
