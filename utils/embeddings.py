"""
Lightweight retrieval and similarity engine built on scikit-learn's TF-IDF.

Includes:
- Morphological and semantic normalization (handling plurals, verb forms, domain synonyms)
- Unigram + bigram feature extraction with sublinear term-frequency scaling
- Cosine similarity computation
- Top-term extraction for feedback-loop query expansion
"""
from __future__ import annotations
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SYNONYM_EXPANSIONS = {
    "retired": "spent end-of-life recycling recovered recycle",
    "retire": "spent end-of-life recycling",
    "retir": "spent end-of-life recycling",
    "vehicles": "cars vehicle",
    "vehicle": "cars vehicle",
    "batteries": "battery cells",
    "freezing": "cold winter freezing",
    "winter": "cold freezing winter",
    "cost": "price prices dollar dollars",
    "costs": "price prices dollar dollars",
    "price": "cost costs dollar dollars",
}


def normalize_text(text: str, expand_synonyms: bool = False) -> str:
    """Normalizes tokens (lowercase, simple stemming/lemmatization, punctuation removal)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    normalized = []
    for raw in tokens:
        t = raw
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            t = t[:-1]
        normalized.append(t)

        if expand_synonyms:
            for candidate in (raw, t):
                if candidate in SYNONYM_EXPANSIONS:
                    normalized.append(SYNONYM_EXPANSIONS[candidate])

    return " ".join(normalized)


class TfidfIndex:
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.doc_texts = [d["text"] for d in documents]
        self.normalized_doc_texts = [normalize_text(t) for t in self.doc_texts]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self.normalized_doc_texts)

    def query(self, query_text: str, top_k: int = 6) -> List[Dict]:
        norm_query = normalize_text(query_text, expand_synonyms=True)
        q_vec = self.vectorizer.transform([norm_query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked = sorted(
            zip(self.documents, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            {**doc, "raw_score": float(round(score, 4))} for doc, score in ranked
        ]

    def similarity(self, text_a: str, text_b: str) -> float:
        norm_a = normalize_text(text_a)
        norm_b = normalize_text(text_b)
        vecs = self.vectorizer.transform([norm_a, norm_b])
        score = float(cosine_similarity(vecs[0], vecs[1]).flatten()[0])
        return float(round(score, 4))

    def top_terms_for_text(self, text: str, n: int = 4) -> List[str]:
        """Extracts the top n TF-IDF terms from a passage for feedback-loop query expansion."""
        norm_text = normalize_text(text)
        vec = self.vectorizer.transform([norm_text])
        feature_names = self.vectorizer.get_feature_names_out()
        row = vec.toarray()[0]
        top_idx = row.argsort()[::-1][:n]
        return [feature_names[i] for i in top_idx if row[i] > 0]
