from __future__ import annotations

import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import feedparser
import httpx

from corroborate.config import PipelineConfig
from corroborate.models import FeedEntry

logger = logging.getLogger(__name__)

# Some feeds reject non-browser clients.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip() if text else ""


def _parse_entry_date(entry: dict) -> datetime | None:
    for field in ("published_parsed", "updated_parsed"):
        parsed = entry.get(field)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _normalize_link(link: str) -> str:
    return link.split("#", 1)[0].split("?", 1)[0].rstrip("/")


class FeedCollectionStage:

    def __init__(self, config: PipelineConfig):
        self._config = config

    def run(self) -> list[FeedEntry]:
        feeds = self._config.load_feeds()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self._config.max_age_hours)

        logger.info("STEP 1: FETCH RSS FEEDS (last %dh)", self._config.max_age_hours)
        logger.info("Cutoff: %s", cutoff.strftime("%Y-%m-%d %H:%M UTC"))

        # Fetch concurrently; map() keeps feeds.yaml order, so duplicate
        # links always resolve to the same (first-listed) feed.
        workers = max(1, min(self._config.feed_workers, len(feeds)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(self._fetch, feeds))

        entries: list[FeedEntry] = []
        seen_links: set[str] = set()
        total_raw = 0
        dropped = {"words": 0, "time": 0, "no_date": 0, "duplicate": 0}

        for fc, parsed, elapsed in fetched:
            if parsed is None:
                continue

            if not parsed.entries:
                logger.warning("  %s — 0 entries (%.1fs)", fc["name"], elapsed)
                continue

            kept = 0
            feed_drops = {"words": 0, "time": 0, "no_date": 0, "duplicate": 0}
            source_type = fc.get("source_type", "editorial")

            for entry in parsed.entries:
                total_raw += 1
                title = _strip_html(entry.get("title", ""))
                summary = _strip_html(entry.get("summary", ""))
                combined = f"{title} {summary}".strip()

                # The word floor protects embedding/clustering from noise.
                # Primary sources skip clustering (they go straight to the
                # digest), and many publish title-only items (ECB, BoJ).
                if (
                    source_type != "primary"
                    and len(combined.split()) < self._config.min_combined_words
                ):
                    feed_drops["words"] += 1
                    continue

                pub_date = _parse_entry_date(entry)
                if pub_date is None:
                    feed_drops["no_date"] += 1
                    continue
                if pub_date < cutoff:
                    feed_drops["time"] += 1
                    continue

                link = entry.get("link", "")
                key = _normalize_link(link)
                if key and key in seen_links:
                    feed_drops["duplicate"] += 1
                    continue
                seen_links.add(key)

                entries.append(FeedEntry(
                    outlet=fc["outlet"],
                    feed_name=fc["name"],
                    title=title,
                    summary=summary,
                    link=link,
                    published=pub_date,
                    source_type=source_type,
                ))
                kept += 1

            for key in dropped:
                dropped[key] += feed_drops[key]

            drop_parts = []
            if feed_drops["words"]:
                drop_parts.append(f"{feed_drops['words']} short")
            if feed_drops["time"]:
                drop_parts.append(f"{feed_drops['time']} old")
            if feed_drops["no_date"]:
                drop_parts.append(f"{feed_drops['no_date']} no date")
            if feed_drops["duplicate"]:
                drop_parts.append(f"{feed_drops['duplicate']} duplicate")
            drop_note = f", dropped: {', '.join(drop_parts)}" if drop_parts else ""
            logger.info("  %s — %d kept (%.1fs%s)", fc["name"], kept, elapsed, drop_note)

        logger.info(
            "Total: %d entries kept from %d raw "
            "(dropped: %d short, %d old, %d no date, %d duplicate)",
            len(entries), total_raw,
            dropped["words"], dropped["time"], dropped["no_date"],
            dropped["duplicate"],
        )

        return entries

    def _fetch(
        self, fc: dict
    ) -> tuple[dict, feedparser.FeedParserDict | None, float]:
        """Download one feed with httpx (certifi CAs, timeout) and parse it."""
        start = time.time()
        try:
            resp = httpx.get(
                fc["url"],
                timeout=self._config.feed_timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        except httpx.HTTPError as e:
            logger.warning(
                "  %s — fetch failed (%.1fs): %s",
                fc["name"], time.time() - start, e.__class__.__name__,
            )
            return fc, None, time.time() - start

        elapsed = time.time() - start
        if resp.status_code >= 400:
            logger.warning("  %s — HTTP %d (%.1fs)", fc["name"], resp.status_code, elapsed)
            return fc, None, elapsed

        parsed = feedparser.parse(
            resp.content,
            response_headers={
                "content-location": str(resp.url),
                "content-type": resp.headers.get("content-type", ""),
            },
        )
        if parsed.bozo and not parsed.entries:
            logger.warning(
                "  %s — unparseable feed (%.1fs): %s",
                fc["name"], elapsed,
                parsed.get("bozo_exception", "").__class__.__name__,
            )
            return fc, None, elapsed

        return fc, parsed, elapsed
