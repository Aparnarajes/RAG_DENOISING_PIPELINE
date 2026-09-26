import math
import os
import unittest
from unittest.mock import patch

from agents.relevance_scoring_agent import RelevanceScoringAgent
from agents.retrieval_agent import RetrievalAgent
from utils.embeddings import SentenceTransformerIndex


class FakeSentenceTransformer:
    """Small deterministic encoder for offline tests of retrieval behavior."""

    GROUPS = {
        "dog": 0,
        "puppy": 0,
        "pet": 0,
        "car": 1,
        "automobile": 1,
        "vehicle": 1,
        "protect": 2,
        "safeguard": 2,
        "coffee": 3,
        "brew": 3,
        "astronomy": 4,
        "galaxy": 4,
        "orbit": 4,
    }

    def __init__(self):
        self.encoded_batches = []

    def encode(
        self,
        sentences,
        *,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        self.encoded_batches.append(list(sentences))
        vectors = []
        for sentence in sentences:
            vector = [0.0] * 5
            for token in sentence.lower().split():
                dimension = self.GROUPS.get(token.strip(".,?!"))
                if dimension is not None:
                    vector[dimension] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


def make_index(documents):
    model = FakeSentenceTransformer()
    return SentenceTransformerIndex(documents, model=model), model


class SemanticRetrievalTests(unittest.TestCase):
    def test_relevant_document_is_retrieved_and_kept(self):
        documents = [
            {"id": "dog-care", "title": "Dog care", "text": "A dog is a loyal pet."},
            {"id": "coffee", "title": "Coffee", "text": "Coffee brewing uses hot water."},
        ]
        index, _ = make_index(documents)
        agent = RelevanceScoringAgent(index, threshold=0.5)

        retrieved = RetrievalAgent(index).retrieve("What helps a dog?", top_k=1)
        scored = agent.score("What helps a dog?", retrieved)

        self.assertEqual([doc["id"] for doc in retrieved], ["dog-care"])
        self.assertEqual(scored[0]["decision"], "KEEP")
        self.assertEqual(scored[0]["document_id"], "dog-care")
        self.assertGreaterEqual(scored[0]["score"], 0.5)

    def test_irrelevant_document_is_removed(self):
        index, _ = make_index([
            {"id": "coffee", "title": "Coffee", "text": "Coffee brewing uses hot water."}
        ])
        agent = RelevanceScoringAgent(index, threshold=0.5)

        scored = agent.score("What helps a dog?", index.query("What helps a dog?"))

        self.assertEqual(scored[0]["decision"], "REMOVE")
        self.assertEqual(agent.filter(scored), [])

    def test_semantically_similar_wording_matches(self):
        index, _ = make_index([
            {
                "id": "car-safety",
                "title": "Vehicle safety",
                "text": "Ways to safeguard an automobile.",
            }
        ])

        retrieved = RetrievalAgent(index).retrieve("How can I protect my car?")

        self.assertEqual(retrieved[0]["id"], "car-safety")
        self.assertGreater(retrieved[0]["raw_score"], 0.8)

    def test_unrelated_document_ranks_below_related_document(self):
        index, _ = make_index([
            {"id": "space", "title": "Space", "text": "Astronomy, galaxy, and orbit."},
            {"id": "dog-care", "title": "Dog care", "text": "A dog is a loyal pet."},
        ])

        retrieved = RetrievalAgent(index).retrieve("What helps a dog?", top_k=2)

        self.assertEqual([doc["id"] for doc in retrieved], ["dog-care", "space"])

    def test_empty_retrieval_returns_no_documents(self):
        index, _ = make_index([])

        self.assertEqual(index.query("Any question?"), [])
        self.assertEqual(RelevanceScoringAgent(index).score("Any question?", []), [])

    def test_low_similarity_is_removed_at_configured_threshold(self):
        index, _ = make_index([
            {"id": "space", "title": "Space", "text": "Astronomy, galaxy, and orbit."}
        ])
        agent = RelevanceScoringAgent(index, threshold=0.8)
        retrieved = RetrievalAgent(index).retrieve("What helps a dog?")

        scored = agent.score("What helps a dog?", retrieved)

        self.assertLess(scored[0]["score"], 0.8)
        self.assertEqual(scored[0]["decision"], "REMOVE")

    def test_embedding_cache_is_reused_for_queries(self):
        documents = [
            {"id": "dog-care", "title": "Dog care", "text": "A dog is a loyal pet."}
        ]
        index, model = make_index(documents)
        document_encoding_calls = sum(len(batch) for batch in model.encoded_batches)

        index.query("What helps a dog?")
        index.query("What helps a pet?")

        self.assertEqual(
            sum(len(batch) for batch in model.encoded_batches),
            document_encoding_calls + 2,
        )
        self.assertEqual(len(index.document_embeddings), 1)

    def test_relevance_threshold_reads_environment(self):
        index, _ = make_index([])
        with patch.dict(os.environ, {"RAG_RELEVANCE_THRESHOLD": "0.62"}):
            agent = RelevanceScoringAgent(index)

        self.assertEqual(agent.threshold, 0.62)

    def test_default_relevance_threshold_is_documented_value(self):
        index, _ = make_index([])

        self.assertEqual(RelevanceScoringAgent(index).threshold, 0.45)

    def test_scoring_documents_without_retrieval_scores_batches_query_embedding(self):
        documents = [
            {"id": "dog-care", "title": "Dog care", "text": "A dog is a loyal pet."},
            {"id": "coffee", "title": "Coffee", "text": "Coffee brewing uses hot water."},
        ]
        index, model = make_index(documents)
        agent = RelevanceScoringAgent(index)
        calls_before = len(model.encoded_batches)

        scored = agent.score("What helps a dog?", documents)

        self.assertEqual(len(scored), 2)
        self.assertEqual(len(model.encoded_batches), calls_before + 1)
        self.assertEqual(len(model.encoded_batches[-1]), 1)

    def test_invalid_threshold_is_rejected(self):
        index, _ = make_index([])

        with self.assertRaises(ValueError):
            RelevanceScoringAgent(index, threshold=1.1)


if __name__ == "__main__":
    unittest.main()
