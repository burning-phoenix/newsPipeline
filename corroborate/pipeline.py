from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager

from nltk.tokenize import sent_tokenize

from corroborate.config import ModelSettings, PipelineConfig
from corroborate.models import FeedEntry, StoryBrief, StoryCluster
from corroborate.providers.agglomerative import AgglomerativeStrategy
from corroborate.providers.openrouter import (
    OpenRouterEmbedding,
    OpenRouterLLM,
    price_ceiling,
)
from corroborate.providers.trafilatura_extractor import TrafilaturaExtractor
from corroborate.stages.brief_synthesis import BriefSynthesisStage
from corroborate.stages.claim_detection import ClaimDetectionStage
from corroborate.stages.clustering import ClusteringStage
from corroborate.stages.embedding import EmbeddingStage
from corroborate.stages.feed_collection import FeedCollectionStage
from corroborate.stages.full_text_fetch import FullTextFetchStage
from corroborate.stages.rendering import RenderingStage
from corroborate.stages.story_analysis import StoryAnalysisStage

logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        logger.info("[%s took %.1fs]", label, time.time() - start)


class Pipeline:
    """Builds the providers and stages, and schedules the work.

    Only two steps must wait for everything before them: clustering (it
    needs every entry's embedding) and the overview (it needs every
    section). The digest depends only on feed collection, and each story's
    chain (fetch -> claims -> analysis -> section) depends only on itself,
    so all of those run concurrently.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config

        self._embedder = OpenRouterEmbedding(
            api_key=config.api_key,
            model=config.embedding_model,
            batch_size=config.embedding_batch_size,
            max_concurrency=config.embedding_concurrency,
        )
        self._ceilings: dict[tuple, tuple[float, float] | None] = {}
        self._llms = {
            "analysis": self._make_llm(config.analysis_llm),
            "writer": self._make_llm(config.writer_llm),
            "overview": self._make_llm(config.overview_llm),
        }
        extractor = TrafilaturaExtractor(min_words=config.min_full_text_words)

        self._feed_collection = FeedCollectionStage(config)
        self._embedding = EmbeddingStage(self._embedder)
        self._clustering = ClusteringStage(AgglomerativeStrategy(), config)
        self._full_text = FullTextFetchStage(extractor)
        self._claim_detection = ClaimDetectionStage(self._embedder, config)
        self._analysis = StoryAnalysisStage(self._llms["analysis"], config)
        self._writer = BriefSynthesisStage(
            self._llms["writer"], self._llms["overview"], config
        )
        self._renderer = RenderingStage()

    def _make_llm(self, settings: ModelSettings) -> OpenRouterLLM:
        fallback = self._make_llm(settings.fallback) if settings.fallback else None
        return OpenRouterLLM(
            api_key=self._config.api_key,
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            max_tokens=settings.max_tokens,
            provider_routing=settings.provider_routing(self._ceiling(settings)),
            max_concurrency=self._config.max_workers,
            fallback=fallback,
        )

    def _ceiling(self, settings: ModelSettings) -> tuple[float, float] | None:
        """Relative price ceiling, looked up once per distinct route."""
        key = (
            settings.model, settings.ceiling_providers,
            tuple(settings.quantizations), tuple(settings.required_parameters()),
            settings.min_throughput,
        )
        if key not in self._ceilings:
            self._ceilings[key] = price_ceiling(
                self._config.api_key, settings.model, settings.ceiling_providers,
                settings.quantizations, settings.required_parameters(),
                settings.min_throughput,
            )
        return self._ceilings[key]

    def run(self) -> str | None:
        """Run the pipeline; return the finished document, or None."""
        with _timed("Step 1 feed collection"):
            entries = self._feed_collection.run()
        if not entries:
            return None

        primary = [e for e in entries if e.source_type == "primary"]
        editorial = [e for e in entries if e.source_type != "primary"]

        # Load the sentence tokenizer once, before threads share it.
        sent_tokenize("Warm up the tokenizer.")

        with ThreadPoolExecutor(max_workers=1) as side:
            digest_future = side.submit(self._digest, primary) if primary else None
            stories = self._run_editorial(editorial) if editorial else []
            digest = digest_future.result() if digest_future else None

        if not stories and not digest:
            return None

        with _timed("Overview"):
            overview = self._writer.write_overview(
                stories + ([digest] if digest else [])
            )
        return self._renderer.render(overview, stories, digest)

    def _digest(self, entries: list[FeedEntry]) -> StoryBrief | None:
        with _timed("Digest"):
            return self._writer.write_digest(entries)

    def _run_editorial(self, entries: list[FeedEntry]) -> list[StoryBrief]:
        with _timed("Step 2 embedding"):
            embeddings = self._embedding.run(entries)

        with _timed("Step 3 clustering"):
            clusters = self._clustering.run(entries, embeddings)

        selected = self._select(clusters)
        if not selected:
            return []

        logger.info(
            "STEPS 4-6: %d STORIES CONCURRENTLY "
            "(fetch -> claims -> analysis -> section)",
            len(selected),
        )
        # Separate pools: story workers block on article fetches, so the
        # fetches must never queue behind them in the same pool.
        with _timed("Steps 4-6 stories"), \
                ThreadPoolExecutor(self._config.max_workers) as fetch_pool, \
                ThreadPoolExecutor(len(selected)) as story_pool:
            futures = [
                story_pool.submit(self._story, rank, cluster, fetch_pool)
                for rank, cluster in enumerate(selected)
            ]
            results = [future.result() for future in futures]

        return [brief for brief in results if brief is not None]

    def _select(self, clusters: list[StoryCluster]) -> list[StoryCluster]:
        """Stories with enough outlets to corroborate anything, best first."""
        eligible = [
            c for c in clusters if len(c.outlets) >= self._config.claim_min_sources
        ]
        if self._config.all_clusters:
            return eligible
        return eligible[: self._config.top_clusters]

    def _story(
        self, rank: int, cluster: StoryCluster, fetch_pool: Executor
    ) -> StoryBrief | None:
        start = time.time()
        try:
            enriched = self._full_text.fetch(cluster, fetch_pool, rank)
            claims = self._claim_detection.detect(enriched, rank)
            if not claims:
                logger.info(
                    "  Cluster %d: dropped — no claim reached %d outlets",
                    rank + 1, self._config.claim_min_sources,
                )
                return None
            notes = self._analysis.analyze(enriched, claims, rank)
            brief = self._writer.write_section(enriched, claims, notes, rank)
        except Exception as e:
            logger.warning(
                "  Cluster %d (%s) failed — %s: %s",
                rank + 1, cluster.headline, e.__class__.__name__, e,
            )
            return None

        logger.info("  Cluster %d done in %.1fs", rank + 1, time.time() - start)
        return brief

    def log_usage(self) -> None:
        logger.info("USAGE embeddings: %s", self._embedder.usage.summary())
        total = self._embedder.usage.cost
        for role, llm in self._llms.items():
            logger.info("USAGE %-10s %s", role + ":", llm.usage.summary())
            total += llm.usage.cost
        logger.info("TOTAL COST: $%.4f", total)
