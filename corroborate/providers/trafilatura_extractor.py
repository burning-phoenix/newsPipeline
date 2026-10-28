from __future__ import annotations

import logging
import time

import trafilatura

from corroborate.providers.base import ContentExtractor

logger = logging.getLogger(__name__)


class TrafilaturaExtractor(ContentExtractor):

    def __init__(self, min_words: int = 50):
        self._min_words = min_words

    def extract(self, url: str) -> str | None:
        start = time.time()

        try:
            downloaded = trafilatura.fetch_url(url)
        except Exception as e:
            elapsed = time.time() - start
            logger.info("    FAIL download (%.1fs) — %s", elapsed, e.__class__.__name__)
            return None

        elapsed_fetch = time.time() - start

        if not downloaded:
            logger.info("    FAIL download (%.1fs)", elapsed_fetch)
            return None

        try:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
        except Exception as e:
            logger.info(
                "    FAIL extract (%.1fs) — %s", elapsed_fetch, e.__class__.__name__
            )
            return None

        if not text or len(text.split()) < self._min_words:
            logger.info("    FAIL extract (%.1fs)", elapsed_fetch)
            return None

        elapsed_total = time.time() - start
        logger.info("    OK %d words (%.1fs)", len(text.split()), elapsed_total)
        return text
