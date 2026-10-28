import unittest
from email.utils import format_datetime
from datetime import datetime, timezone

import feedparser

from corroborate.config import PipelineConfig
from corroborate.stages.feed_collection import FeedCollectionStage, _strip_html

NOW = format_datetime(datetime.now(timezone.utc))
# Comfortably above the 15-word editorial floor once joined with a title.
LONG_SUMMARY = (
    "Officials confirmed the details of the event in a statement "
    "released on Tuesday afternoon in the capital."
)


def rss(*items: tuple[str, str, str]) -> str:
    body = "".join(
        f"<item><title>{title}</title><link>{link}</link>"
        f"<description>{summary}</description><pubDate>{NOW}</pubDate></item>"
        for title, link, summary in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{body}</channel></rss>'


class StubFeedStage(FeedCollectionStage):
    """Serves canned feed XML instead of fetching over the network."""

    def __init__(self, feeds: list[dict], xml_by_url: dict[str, str]):
        config = PipelineConfig()
        config.load_feeds = lambda: feeds
        super().__init__(config)
        self._xml = xml_by_url

    def _fetch(self, fc):
        return fc, feedparser.parse(self._xml[fc["url"]]), 0.0


def feed(name, outlet, url, source_type="editorial"):
    return {"name": name, "outlet": outlet, "url": url, "source_type": source_type}


class FeedCollectionTest(unittest.TestCase):

    def test_strip_html_unescapes_entities(self):
        self.assertEqual(
            _strip_html("<p>Korea&apos;s &amp; Japan&#8217;s &#8230;</p>"),
            "Korea's & Japan’s …",
        )

    def test_same_link_from_two_feeds_is_kept_once(self):
        stage = StubFeedStage(
            [feed("Guardian World", "The Guardian", "a"),
             feed("Guardian US", "The Guardian", "b")],
            {
                "a": rss(("Story", "https://g.com/story?CMP=rss", LONG_SUMMARY)),
                "b": rss(("Story", "https://g.com/story/", LONG_SUMMARY)),
            },
        )
        entries = stage.run()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].feed_name, "Guardian World")

    def test_word_floor_applies_to_editorial_only(self):
        stage = StubFeedStage(
            [feed("News", "Outlet", "e"),
             feed("ECB", "ECB", "p", source_type="primary")],
            {
                "e": rss(("Short headline", "https://n.com/1", "")),
                "p": rss(("ECB publishes statistics", "https://ecb.eu/1", "")),
            },
        )
        entries = stage.run()
        self.assertEqual([e.outlet for e in entries], ["ECB"])


if __name__ == "__main__":
    unittest.main()
