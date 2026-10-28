from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """Embed a list of texts into vectors.

        Returns (embeddings_array, total_tokens_used).
        """

    @abstractmethod
    def cost_per_million_tokens(self) -> float:
        """Price in USD per million input tokens."""


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self, system_prompt: str, user_prompt: str, *, json_mode: bool = False
    ) -> dict:
        """Send a chat completion request.

        With json_mode, the model is asked to return a single JSON object.
        Returns dict with keys: 'content', 'prompt_tokens', 'completion_tokens'.
        Raises if the answer is empty or was truncated.
        """


class ContentExtractor(ABC):
    @abstractmethod
    def extract(self, url: str) -> str | None:
        """Fetch and extract article text from a URL.

        Returns extracted text, or None if extraction failed.
        """


class ClusteringStrategy(ABC):
    @abstractmethod
    def fit(self, distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
        """Cluster items based on a precomputed distance matrix.

        Returns array of cluster labels.
        """
