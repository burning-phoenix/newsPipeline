from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class FeedEntry:
    outlet: str
    feed_name: str
    title: str
    summary: str
    link: str
    published: datetime
    source_type: str = "editorial"

    @property
    def combined(self) -> str:
        return f"{self.title} {self.summary}".strip()


@dataclass
class StoryCluster:
    entries: list[FeedEntry]
    embeddings: np.ndarray
    headline: str

    @property
    def outlets(self) -> set[str]:
        return {e.outlet for e in self.entries}


@dataclass
class Article:
    entry: FeedEntry
    full_text: str | None = None

    @property
    def text(self) -> str:
        if self.full_text:
            return self.full_text
        return self.entry.combined

    @property
    def source_label(self) -> str:
        return "full-text" if self.full_text else "title+summary only"


@dataclass
class EnrichedCluster:
    headline: str
    articles: list[Article]

    @property
    def outlets(self) -> set[str]:
        return {a.entry.outlet for a in self.articles}

    @property
    def source_type(self) -> str:
        types = {a.entry.source_type for a in self.articles}
        if types == {"primary"}:
            return "primary"
        if types == {"editorial"}:
            return "editorial"
        return "mixed"


@dataclass
class Claim:
    anchor_text: str
    confirming_outlets: set[str] = field(default_factory=set)

    @property
    def source_count(self) -> int:
        return len(self.confirming_outlets)


@dataclass
class Discrepancy:
    """One way the outlets' accounts differ.

    kind is "conflict" (statements that cannot both be true), "figures"
    (different numbers, times or dates for the same thing), or "unique"
    (a significant detail only one or two outlets report).
    """
    topic: str
    positions: dict[str, str]  # outlet -> what that outlet reports
    kind: str = "conflict"


@dataclass
class StoryNotes:
    """The analyst's notes on one story, already checked in code."""
    framing: dict[str, str] = field(default_factory=dict)  # outlet -> angle
    discrepancies: list[Discrepancy] = field(default_factory=list)


@dataclass
class StoryBrief:
    headline: str
    outlets: list[str]
    claims: list[Claim]
    brief_text: str  # the finished markdown section
