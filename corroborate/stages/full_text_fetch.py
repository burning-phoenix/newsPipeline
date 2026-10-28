from __future__ import annotations

import logging
from concurrent.futures import Executor

from corroborate.models import Article, EnrichedCluster, FeedEntry, StoryCluster
from corroborate.providers.base import ContentExtractor

logger = logging.getLogger(__name__)


class FullTextFetchStage:

    def __init__(self, extractor: ContentExtractor):
        self._extractor = extractor

    def fetch(
        self, cluster: StoryCluster, pool: Executor, rank: int = 0
    ) -> EnrichedCluster:
        """Fetch one story's articles on a shared pool.

        Articles whose extraction fails keep title+summary as their text.
        """
        futures = [pool.submit(self._extract, entry) for entry in cluster.entries]
        articles = [
            Article(entry=entry, full_text=future.result())
            for entry, future in zip(cluster.entries, futures)
        ]

        succeeded = sum(1 for a in articles if a.full_text)
        failed = [a.entry.outlet for a in articles if not a.full_text]
        logger.info(
            "  Cluster %d: %s — fetched %d/%d",
            rank + 1, cluster.headline, succeeded, len(articles),
        )
        if failed:
            logger.info(
                "  Cluster %d: fallback to title+summary: %s",
                rank + 1, ", ".join(failed),
            )

        return EnrichedCluster(headline=cluster.headline, articles=articles)

    def _extract(self, entry: FeedEntry) -> str | None:
        try:
            return self._extractor.extract(entry.link)
        except Exception as e:
            logger.info(
                "    [%-20s] FAIL — %s", entry.outlet, e.__class__.__name__
            )
            return None
