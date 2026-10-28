import math
import unittest
from datetime import datetime, timezone

import numpy as np

from corroborate.config import PipelineConfig
from corroborate.models import Article, EnrichedCluster, FeedEntry
from corroborate.providers.base import EmbeddingProvider
from corroborate.stages.claim_detection import ClaimDetectionStage


def unit(angle_deg: float, plane: int = 0) -> list[float]:
    """A unit vector at angle_deg inside one of two orthogonal planes.

    Cosine similarity between two vectors in the same plane is
    cos(angle difference); vectors in different planes are orthogonal.
    """
    vec = [0.0, 0.0, 0.0, 0.0]
    rad = math.radians(angle_deg)
    vec[2 * plane], vec[2 * plane + 1] = math.cos(rad), math.sin(rad)
    return vec


class FakeEmbedding(EmbeddingProvider):
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        return np.array([self._vectors[t] for t in texts]), 0

    def cost_per_million_tokens(self) -> float:
        return 0.0


def make_cluster(sentences_by_outlet: dict[str, list[str]]) -> EnrichedCluster:
    now = datetime.now(timezone.utc)
    articles = [
        Article(
            entry=FeedEntry(outlet, outlet, "title", "summary", f"https://{i}", now),
            full_text=" ".join(sentences),
        )
        for i, (outlet, sentences) in enumerate(sentences_by_outlet.items())
    ]
    return EnrichedCluster(headline="Test story", articles=articles)


# Sentences need at least min_sentence_words (8) words to be considered.
A_X = "The bridge over the river collapsed early on Tuesday morning."
A_Y = "Early on Tuesday the river bridge collapsed into the water below."
A_Z = "A bridge spanning the river fell down in the early Tuesday hours."
A_W = "Tuesday morning saw the collapse of the bridge across the river."
B_X = "Officials said that twelve people were rescued from the water afterwards."
B_Y = "Rescue crews pulled twelve people out of the river after the collapse."
B_Z = "Twelve survivors were recovered from the river by emergency rescue crews."


class ClaimDetectionTest(unittest.TestCase):

    def detect(self, sentences_by_outlet, vectors):
        stage = ClaimDetectionStage(FakeEmbedding(vectors), PipelineConfig())
        return stage.detect(make_cluster(sentences_by_outlet))

    def test_three_outlets_make_a_claim_two_do_not(self):
        claims = self.detect(
            {"X": [A_X], "Y": [A_Y], "Z": [B_Z]},
            {A_X: unit(0), A_Y: unit(10), B_Z: unit(0, plane=1)},
        )
        self.assertEqual(claims, [])

        claims = self.detect(
            {"X": [A_X], "Y": [A_Y], "Z": [A_Z]},
            {A_X: unit(0), A_Y: unit(10), A_Z: unit(20)},
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].confirming_outlets, {"X", "Y", "Z"})

    def test_same_outlet_does_not_count_twice(self):
        claims = self.detect(
            {"X": [A_X, A_W], "Y": [A_Y]},
            {A_X: unit(0), A_W: unit(5), A_Y: unit(10)},
        )
        self.assertEqual(claims, [])

    def test_claim_is_worded_by_the_most_central_sentence(self):
        # X comes first but sits at the edge (0°); Y (20°) is in the middle.
        claims = self.detect(
            {"X": [A_X], "Y": [A_Y], "Z": [A_Z]},
            {A_X: unit(0), A_Y: unit(20), A_Z: unit(40)},
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].anchor_text, A_Y)

    def test_overlapping_matches_merge_and_pool_outlets(self):
        # Anchor X (0°) matches Y, Z; W (45°) is too far from X (cos 45° < 0.75)
        # so W anchors a second raw claim {W, Y, Z}. Both describe the same
        # fact, so they merge into one claim confirmed by all four outlets.
        claims = self.detect(
            {"X": [A_X], "Y": [A_Y], "Z": [A_Z], "W": [A_W]},
            {A_X: unit(0), A_Y: unit(10), A_Z: unit(20), A_W: unit(45)},
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].confirming_outlets, {"X", "Y", "Z", "W"})

    def test_distinct_facts_stay_separate(self):
        claims = self.detect(
            {"X": [A_X, B_X], "Y": [A_Y, B_Y], "Z": [A_Z, B_Z]},
            {
                A_X: unit(0), A_Y: unit(10), A_Z: unit(20),
                B_X: unit(0, plane=1), B_Y: unit(10, plane=1), B_Z: unit(20, plane=1),
            },
        )
        self.assertEqual(len(claims), 2)
        self.assertEqual({c.anchor_text for c in claims}, {A_Y, B_Y})


if __name__ == "__main__":
    unittest.main()
