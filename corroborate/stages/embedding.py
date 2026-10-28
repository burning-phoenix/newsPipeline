from __future__ import annotations

import logging

import numpy as np

from corroborate.models import FeedEntry
from corroborate.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingStage:

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider

    def run(self, entries: list[FeedEntry]) -> np.ndarray:
        logger.info("STEP 2: EMBED (%d entries)", len(entries))

        texts = [e.combined for e in entries]
        embeddings, total_tokens = self._provider.embed(texts)

        cost = total_tokens * self._provider.cost_per_million_tokens() / 1_000_000
        logger.info(
            "Embeddings: %d vectors, %d dimensions — %d tokens, $%.6f",
            embeddings.shape[0], embeddings.shape[1], total_tokens, cost,
        )

        return embeddings
