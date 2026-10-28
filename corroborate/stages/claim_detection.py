from __future__ import annotations

import logging

import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from corroborate.config import PipelineConfig
from corroborate.models import Claim, EnrichedCluster
from corroborate.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ClaimDetectionStage:

    def __init__(self, provider: EmbeddingProvider, config: PipelineConfig):
        self._provider = provider
        self._sim_threshold = config.claim_similarity_threshold
        self._min_sources = config.claim_min_sources
        self._min_sentence_words = config.min_sentence_words

    def detect(self, cluster: EnrichedCluster, rank: int = 0) -> list[Claim]:
        """Find claims in one story confirmed by claim_min_sources outlets."""
        label = f"Cluster {rank + 1}"

        if cluster.source_type == "primary":
            logger.info("  %s: skipped — primary source", label)
            return []

        sentences, outlets = self._split_sentences(cluster)
        logger.info(
            "  %s: %s — %d articles, %d outlets, %d sentences",
            label, cluster.headline, len(cluster.articles),
            len(cluster.outlets), len(sentences),
        )

        if len(sentences) < 3:
            logger.info("  %s: skipped — too few sentences", label)
            return []

        try:
            embeddings, _ = self._provider.embed(sentences)
        except Exception as e:
            logger.warning(
                "  %s: embedding failed — %s: %s. Skipping cluster.",
                label, e.__class__.__name__, e,
            )
            return []

        found = self._find_claims(sentences, outlets, embeddings)
        claims = self._deduplicate(found, embeddings)

        logger.info(
            "  %s: %d corroborated claims (from %d raw, deduplicated)",
            label, len(claims), len(found),
        )
        for i, claim in enumerate(claims):
            logger.info(
                "    CLAIM %d — %d sources (%s): %.200s",
                i + 1, claim.source_count,
                ", ".join(sorted(claim.confirming_outlets)),
                claim.anchor_text,
            )
        return claims

    def _split_sentences(
        self, cluster: EnrichedCluster
    ) -> tuple[list[str], list[str]]:
        sentences: list[str] = []
        outlets: list[str] = []
        for article in cluster.articles:
            for sent in sent_tokenize(article.text):
                sent = sent.strip()
                if len(sent.split()) >= self._min_sentence_words:
                    sentences.append(sent)
                    outlets.append(article.entry.outlet)
        return sentences, outlets

    def _find_claims(
        self,
        sentences: list[str],
        outlets: list[str],
        embeddings: np.ndarray,
    ) -> list[tuple[int, Claim]]:
        """Return (anchor sentence index, claim) pairs."""
        sim_matrix = cosine_similarity(embeddings)
        used: set[int] = set()
        found: list[tuple[int, Claim]] = []

        for i in range(len(sentences)):
            if i in used:
                continue

            matching_outlets = {outlets[i]}
            members = [i]

            for j in range(len(sentences)):
                if i == j or outlets[j] == outlets[i]:
                    continue
                if sim_matrix[i, j] >= self._sim_threshold:
                    matching_outlets.add(outlets[j])
                    members.append(j)

            if len(matching_outlets) >= self._min_sources:
                # Word the claim with the sentence closest to all the others,
                # not whichever outlet happens to come first in feeds.yaml.
                anchor = max(members, key=lambda m: sim_matrix[m, members].mean())
                found.append((anchor, Claim(
                    anchor_text=sentences[anchor],
                    confirming_outlets=matching_outlets,
                )))
                used.update(members)

        return found

    def _deduplicate(
        self,
        found: list[tuple[int, Claim]],
        embeddings: np.ndarray,
    ) -> list[Claim]:
        """Merge claims whose anchors say the same thing, pooling outlets."""
        kept: list[tuple[int, Claim]] = []

        for anchor, claim in found:
            for kept_anchor, kept_claim in kept:
                sim = cosine_similarity(
                    embeddings[anchor].reshape(1, -1),
                    embeddings[kept_anchor].reshape(1, -1),
                )[0, 0]
                if sim >= self._sim_threshold:
                    kept_claim.confirming_outlets |= claim.confirming_outlets
                    break
            else:
                kept.append((anchor, claim))

        return [claim for _, claim in kept]
