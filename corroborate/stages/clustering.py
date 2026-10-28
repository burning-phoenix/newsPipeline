from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from corroborate.config import PipelineConfig
from corroborate.models import FeedEntry, StoryCluster
from corroborate.providers.base import ClusteringStrategy

logger = logging.getLogger(__name__)


class ClusteringStage:

    def __init__(self, strategy: ClusteringStrategy, config: PipelineConfig):
        self._strategy = strategy
        self._thresholds = config.clustering_thresholds

    def run(
        self, entries: list[FeedEntry], embeddings: np.ndarray
    ) -> list[StoryCluster]:
        logger.info("STEP 3: CLUSTERING (threshold sweep)")

        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = np.clip(1.0 - sim_matrix, 0, None)
        np.fill_diagonal(distance_matrix, 0)

        best_threshold = None
        best_labels = None
        best_multi = 0

        for threshold in self._thresholds:
            labels = self._strategy.fit(distance_matrix, threshold)
            multi_source = self._count_multi_source(entries, labels)

            logger.info(
                "  threshold=%.2f — %d clusters, %d multi-source, %d singletons",
                threshold, len(set(labels)), multi_source,
                sum(1 for c in Counter(labels).values() if c == 1),
            )

            if multi_source > best_multi:
                best_multi = multi_source
                best_threshold = threshold
                best_labels = labels

        if best_labels is None:
            logger.info("No multi-source clusters found at any threshold")
            return []

        logger.info("Best: threshold=%.2f (%d multi-source clusters)",
                     best_threshold, best_multi)

        clusters = self._build_clusters(entries, embeddings, best_labels)
        self._log_clusters(clusters)
        return clusters

    def _count_multi_source(
        self, entries: list[FeedEntry], labels: np.ndarray
    ) -> int:
        count = 0
        for cluster_id in set(labels):
            outlets = {
                entries[i].outlet for i, l in enumerate(labels) if l == cluster_id
            }
            if len(outlets) >= 2:
                count += 1
        return count

    def _build_clusters(
        self,
        entries: list[FeedEntry],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> list[StoryCluster]:
        cluster_map: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            cluster_map.setdefault(int(label), []).append(i)

        clusters: list[StoryCluster] = []
        for member_indices in cluster_map.values():
            members = [entries[i] for i in member_indices]
            outlets = {m.outlet for m in members}
            if len(outlets) < 2:
                continue

            member_embeddings = embeddings[member_indices]
            centroid = member_embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(member_embeddings, centroid).flatten()
            headline = members[int(sims.argmax())].title

            clusters.append(StoryCluster(
                entries=members,
                embeddings=member_embeddings,
                headline=headline,
            ))

        # Rank by independent outlets first: five feeds from one outlet
        # are not five sources.
        clusters.sort(key=lambda c: (-len(c.outlets), -len(c.entries)))
        return clusters

    def _log_clusters(self, clusters: list[StoryCluster]) -> None:
        for cluster in clusters:
            intra_sim = cosine_similarity(cluster.embeddings)
            np.fill_diagonal(intra_sim, 0)
            upper = intra_sim[np.triu_indices(len(cluster.entries), k=1)]
            avg_sim = float(upper.mean()) if len(upper) > 0 else 0.0

            logger.info(
                "  Story: %s (%d articles, %d outlets, avg sim %.3f)",
                cluster.headline, len(cluster.entries),
                len(cluster.outlets), avg_sim,
            )
            centroid = cluster.embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(cluster.embeddings, centroid).flatten()
            for idx, entry in enumerate(cluster.entries):
                logger.info(
                    "    [%-20s] (sim=%.3f) %s",
                    entry.outlet, sims[idx], entry.title,
                )
