from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative

from corroborate.providers.base import ClusteringStrategy


class AgglomerativeStrategy(ClusteringStrategy):

    def __init__(self, linkage: str = "average"):
        self._linkage = linkage

    def fit(self, distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
        clustering = SklearnAgglomerative(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage=self._linkage,
        )
        return clustering.fit_predict(distance_matrix)
