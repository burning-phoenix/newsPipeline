from __future__ import annotations

import logging

from corroborate.attribution import format_attribution, unverified_attributions
from corroborate.config import PipelineConfig
from corroborate.models import (
    Claim,
    EnrichedCluster,
    FeedEntry,
    StoryBrief,
    StoryNotes,
)
from corroborate.providers.base import LLMProvider

logger = logging.getLogger(__name__)

DIGEST_HEADING = "## Economic & Monetary Policy"


def _strip_leading_title(text: str) -> str:
    """Drop a leading H1 if the model wrote one despite being told not to."""
    stripped = text.lstrip()
    if not stripped.startswith("# "):
        return stripped
    _, _, rest = stripped.partition("\n")
    return rest.lstrip()


def _ensure_heading(text: str, fallback: str) -> tuple[str, str]:
    """Return (headline, section) with the section starting at a ## heading."""
    text = _strip_leading_title(text)
    first, _, rest = text.partition("\n")
    if first.startswith("## "):
        return first[3:].strip(), text
    return fallback, f"## {fallback}\n\n{text}"


EXAMPLE_SECTION = """## Portugal: Nationwide General Strike Disrupts Services

Workers across Portugal walked off the job on Tuesday in a nationwide general \
strike that disrupted public transport, schools and non-emergency hospital \
services (France 24, BBC News, Deutsche Welle). The strike was called by the \
country's two largest union federations, CGTP and UGT, in protest against \
proposed labour reforms (Deutsche Welle, France 24 and others). Lisbon's metro \
ran at roughly 30% capacity during the morning rush (BBC News, Euronews, France 24).

Deutsche Welle framed the walkout as the largest labour action in Portugal in \
over a decade, while Brussels Morning focused on its economic cost, writing that \
it "exposed fault lines in Portugal's post-austerity recovery." Neither outlet \
endorsed or condemned the strike.

The outlets disagreed on the strike's reach: France 24 reported that all public \
services were affected, while BBC News limited the disruption to transport, \
health and education. Turnout estimates for the Lisbon march also varied, from \
20,000 according to Euronews to 50,000 according to Deutsche Welle. Only \
Brussels Morning reported that the government had offered to reopen talks."""

SECTION_INSTRUCTIONS = """## TASK
Write this story's section of the daily brief, in markdown:

- Begin with a "## " heading: a short, neutral headline, not any outlet's framing.
- Paragraph 1 (3-5 sentences): what happened, told only through the verified \
claims. Use the most important ones; you need not use them all. Right after \
each sentence that states a claim, put that claim's attribution exactly as \
given above. If one sentence states two claims, put both attributions in one \
pair of parentheses separated by a semicolon. If two claims state the same \
fact, state it once. You may paraphrase claims, but add no fact, number, \
date, name or quote that the claims do not contain.
- Paragraph 2: how the outlets framed the story, from the framing notes. Name \
each outlet in the sentence itself ("BBC News emphasized..."), never in parentheses.
- Paragraph 3 (2-4 sentences), always: where the accounts differ, from the \
list above, naming each outlet in the sentence. Describe each item as what it \
is. A [conflict] is a disagreement ("BBC News reported ..., while Al Jazeera \
said ..."). [figures] go side by side or as a range ("estimates ranged from \
... to ..."). A [unique] detail is reported by that outlet alone ("Only CBS \
News reported that ..."). Call something a disagreement only if it is marked \
[conflict]. Skip an item if its positions actually say the same thing or are \
plainly about different things. If nothing is left, write one sentence saying \
the outlets' accounts of these facts are consistent.
- 200-350 words, flowing prose, no bullet points. Calm, measured tone.
- Use parentheses only for the attribution strings."""

DIGEST_INSTRUCTIONS = f"""## TASK
Write the brief's section on these releases, in markdown:

- Begin with exactly this heading: {DIGEST_HEADING}
- Flowing prose organized by theme, 2-4 short paragraphs, naming institutions \
and speakers naturally. No bullet points.
- Use only what the releases above state. Add no figures, dates or background \
of your own. The releases are from the past 24 hours.
- 150-300 words. Calm, measured tone."""

OVERVIEW_SYSTEM = (
    "You write the opening lines of a daily news brief. Your tone is calm, "
    "measured and unhurried. You respect the reader's time."
)

OVERVIEW_INSTRUCTIONS = """## TASK
Write a 2-3 sentence overview of today's brief, drawn only from the section \
openings above. Mention the most significant stories. Add no day, date, \
number, name or detail that is not stated in them. No heading, no \
parentheses, no bullet points: just the paragraph."""


class BriefSynthesisStage:
    """Writes the prose the reader sees: story sections, digest, overview.

    The section writer never sees the articles, only verified claims (with
    attribution strings fixed in code) and the analyst's checked notes, so
    it cannot bring in unverified facts.
    """

    def __init__(
        self,
        writer: LLMProvider,
        overview_writer: LLMProvider,
        config: PipelineConfig,
    ):
        self._writer = writer
        self._overview_writer = overview_writer
        self._config = config

    def _system_prompt(self) -> str:
        return (
            self._config.load_prompt("constitution.md")
            + "\n\n---\n\n"
            + self._config.load_prompt("brief_synthesizer/soul.md")
        )

    def write_section(
        self,
        cluster: EnrichedCluster,
        claims: list[Claim],
        notes: StoryNotes,
        rank: int = 0,
    ) -> StoryBrief | None:
        # Most-corroborated first; the stable sort keeps discovery order on ties.
        claims = sorted(claims, key=lambda c: -c.source_count)[
            : self._config.max_claims_per_story
        ]
        attributions = [format_attribution(claim) for claim in claims]
        user_prompt = self._build_section_prompt(cluster, claims, attributions, notes)

        try:
            result = self._writer.complete(self._system_prompt(), user_prompt)
        except Exception as e:
            logger.warning(
                "  Cluster %d (%s): writing failed — %s: %s. Skipping.",
                rank + 1, cluster.headline, e.__class__.__name__, e,
            )
            return None

        headline, section = _ensure_heading(result["content"], cluster.headline)

        unverified = unverified_attributions(
            section, set(attributions), cluster.outlets
        )
        if unverified:
            logger.warning(
                "  Cluster %d: %d attribution(s) not from a verified claim: %s",
                rank + 1, len(unverified), "; ".join(unverified),
            )

        return StoryBrief(
            headline=headline,
            outlets=sorted(cluster.outlets),
            claims=claims,
            brief_text=section,
        )

    def write_digest(self, entries: list[FeedEntry]) -> StoryBrief | None:
        outlets = sorted({e.outlet for e in entries})
        logger.info(
            "DIGEST: %d entries from %s", len(entries), ", ".join(outlets)
        )

        try:
            result = self._writer.complete(
                self._system_prompt(), self._build_digest_prompt(entries)
            )
        except Exception as e:
            logger.warning(
                "  Digest failed — %s: %s. Skipping.", e.__class__.__name__, e,
            )
            return None

        _, section = _ensure_heading(result["content"], DIGEST_HEADING[3:])
        return StoryBrief(
            headline=DIGEST_HEADING[3:],
            outlets=outlets,
            claims=[],
            brief_text=section,
        )

    def write_overview(self, briefs: list[StoryBrief]) -> str | None:
        """2-3 sentence overview from each section's heading and opening."""
        parts = ["## SECTION OPENINGS\n"]
        for brief in briefs:
            _, _, body = brief.brief_text.partition("\n")
            opening = body.strip().split("\n\n", 1)[0]
            parts.append(f"### {brief.headline}\n{opening}\n")
        parts.append(OVERVIEW_INSTRUCTIONS)

        try:
            result = self._overview_writer.complete(OVERVIEW_SYSTEM, "\n".join(parts))
        except Exception as e:
            logger.warning(
                "  Overview failed — %s: %s. Rendering without it.",
                e.__class__.__name__, e,
            )
            return None
        return _strip_leading_title(result["content"])

    def _build_section_prompt(
        self,
        cluster: EnrichedCluster,
        claims: list[Claim],
        attributions: list[str],
        notes: StoryNotes,
    ) -> str:
        parts: list[str] = [
            f"## VERIFIED CLAIMS (each confirmed by at least "
            f"{self._config.claim_min_sources} independent outlets)\n",
        ]
        for i, (claim, attribution) in enumerate(zip(claims, attributions)):
            parts.append(f'{i + 1}. "{claim.anchor_text}"')
            parts.append(f"   Attribution: {attribution}")
        parts.append("")

        parts.append("## HOW OUTLETS FRAMED IT\n")
        if notes.framing:
            for outlet, angle in notes.framing.items():
                parts.append(f"- {outlet}: {angle}")
        else:
            parts.append("No framing notes. Keep paragraph 2 to one short sentence, or omit it.")
        parts.append("")

        parts.append("## WHERE ACCOUNTS DIFFER\n")
        if notes.discrepancies:
            for d in notes.discrepancies:
                sides = "; ".join(
                    f'{outlet} reports "{says}"' for outlet, says in d.positions.items()
                )
                if d.kind == "unique":
                    sides = "only " + sides
                parts.append(f"- [{d.kind}] {d.topic}: {sides}")
        else:
            parts.append(
                "Nothing listed: the outlets' accounts of these facts match."
            )
        parts.append("")

        parts.append(
            f"## OUTLETS COVERING THIS STORY\n{', '.join(sorted(cluster.outlets))}\n"
        )
        parts.append("## EXAMPLE OF A GOOD SECTION\n")
        parts.append(EXAMPLE_SECTION)
        parts.append("")
        parts.append(SECTION_INSTRUCTIONS)
        return "\n".join(parts)

    def _build_digest_prompt(self, entries: list[FeedEntry]) -> str:
        parts: list[str] = ["## INSTITUTIONAL SOURCES (primary, authoritative)\n"]
        parts.append(
            "The following are releases, speeches, papers, and data from "
            "central banks, international financial institutions, and official "
            "agencies. These institutions are the source of truth: no "
            "cross-source verification is needed.\n"
        )

        by_outlet: dict[str, list[FeedEntry]] = {}
        for entry in entries:
            by_outlet.setdefault(entry.outlet, []).append(entry)

        for outlet in sorted(by_outlet):
            parts.append(f"### {outlet}\n")
            for entry in by_outlet[outlet]:
                parts.append(f"**{entry.title}**")
                if entry.summary:
                    parts.append(entry.summary)
                parts.append("")

        parts.append(DIGEST_INSTRUCTIONS)
        return "\n".join(parts)
