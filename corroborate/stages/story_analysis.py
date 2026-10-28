from __future__ import annotations

import json
import logging
import re

from corroborate.config import PipelineConfig
from corroborate.models import Claim, Discrepancy, EnrichedCluster, StoryNotes
from corroborate.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# A "position" that only says an outlet left something out is not a
# statement the outlet made ("did not name the pilot", "unspecified").
# Such positions are dropped; what the other outlets report can still
# stand as a "unique" detail.
_OMISSION = re.compile(
    r"\b(did|does|do)\s+not\s+(mention|name|report|specify|give|provide|"
    r"identify|say|include|state|disclose|claim|note|address|describe|cite)\b"
    # An inference about what an outlet means is not a statement it made.
    r"|\bimpl(y|ies|ying)\b"
    r"|\b(offered|gave|provided|reported|gives|offers)\s+no\b"
    r"|\bno\s+(figure|number|name|total|details?)\b"
    r"|\bnot\s+(yet\s+)?(been\s+)?(revealed|released|disclosed|confirmed)\b"
    r"|\b(unnamed|unspecified|not\s+(named|specified|mentioned|reported|identified))\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")

# Conflicts first, then differing figures, then details only some report.
KIND_ORDER = {"conflict": 0, "figures": 1, "unique": 2}
MAX_DIFFERENCES = 4
MAX_UNIQUE = 2

INSTRUCTIONS = """## TASK
Take notes on the articles above for the writer, who will not see them.

Return a single JSON object, and nothing else, in this shape:
{
  "framing": [
    {"outlet": "<outlet name exactly as given>", "angle": "<one sentence: what it emphasizes, leads with or leaves out; tone; any loaded words, quoted>"}
  ],
  "differences": [
    {"kind": "conflict | figures | unique",
     "topic": "<what the difference is about, a few words>",
     "positions": [
       {"outlet": "<outlet name>", "says": "<what this outlet reports, in its own words where possible>"}
     ]}
  ]
}

framing: at most one entry per outlet; skip outlets that only ran a headline.

differences: compare the articles carefully and list where their accounts differ.
- "conflict": two or more outlets state things that cannot both be true
  (who did something, whether it happened, its cause, the order of events).
- "figures": two or more outlets give different numbers, amounts, times or
  dates for the same thing at the same point in time (death tolls, crowd
  sizes, vote counts, percentages, how long before or after something).
- "unique": a significant detail that only one or two outlets report. List
  only the outlets that report it.
Each position must be something that outlet actually states. Never write a
position like "did not mention X"; leave that outlet out instead. Never infer
what an outlet "implies". Different quotes from the same speaker are not a
conflict. The verified claims above are settled; do not list them.
Return an empty list only if the accounts truly match."""


def _parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating a surrounding code fence."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _normalize(text: str) -> str:
    return re.sub(r"[^\w.%$]+", " ", text.lower()).strip()


def _match_outlet(name: object, outlets: set[str]) -> str | None:
    """Map the model's outlet name onto a cluster outlet, or None.

    Models write "The New York Times" or "Al Jazeera English" for
    "New York Times" / "Al Jazeera"; accept those, but only when exactly
    one cluster outlet fits.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    if name in outlets:
        return name
    wanted = name.strip().lower()
    matches = [
        outlet for outlet in outlets
        if outlet.lower() == wanted
        or outlet.lower() in wanted
        or wanted in outlet.lower()
    ]
    return matches[0] if len(matches) == 1 else None


def _raw_differences(raw: dict) -> list[dict]:
    """The model's differences list (older prompts called it discrepancies)."""
    value = raw.get("differences")
    if value is None:
        value = raw.get("discrepancies")
    return _items(value, "topic", "positions")


def _items(value: object, key: str, value_key: str) -> list[dict]:
    """Accept a list of objects, or a {outlet: text} map, as a list."""
    if isinstance(value, dict):
        return [{key: k, value_key: v} for k, v in value.items()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _is_rounding(values: list[str]) -> bool:
    """True when every position is the same single figure within 2%."""
    figures = []
    for value in values:
        numbers = _NUMBER.findall(value)
        if len(numbers) != 1:
            return False
        figures.append(float(numbers[0].replace(",", "").rstrip(".")))
    low, high = min(figures), max(figures)
    return low > 0 and high <= low * 1.02


def _truncate_words(text: str, limit: int) -> str:
    """Keep the first `limit` words, marking the cut so the model knows."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " […]"


class StoryAnalysisStage:
    """Reads a story's full articles; notes framing and real disagreements.

    This is the input-heavy step (every article's text), and its output is
    notes for the writer rather than prose a reader sees, so it runs on the
    cheaper model. The notes are checked in code before the writer gets them.
    """

    def __init__(self, llm: LLMProvider, config: PipelineConfig):
        self._llm = llm
        self._config = config

    def analyze(
        self, cluster: EnrichedCluster, claims: list[Claim], rank: int = 0
    ) -> StoryNotes:
        system_prompt = (
            self._config.load_prompt("constitution.md")
            + "\n\n---\n\n"
            + self._config.load_prompt("story_analyst/soul.md")
        )
        user_prompt = self._build_prompt(cluster, claims)

        try:
            result = self._llm.complete(system_prompt, user_prompt, json_mode=True)
            raw = _parse_json(result["content"])
        except Exception as e:
            logger.warning(
                "  Cluster %d: analysis failed — %s: %s. Writing from claims only.",
                rank + 1, e.__class__.__name__, e,
            )
            return StoryNotes()

        notes = self._check(raw, cluster.outlets)
        kinds = [d.kind for d in notes.discrepancies]
        logger.info(
            "  Cluster %d: analysis — %d of %d framing notes, %d of %d "
            "differences kept (%d conflict, %d figures, %d unique)",
            rank + 1,
            len(notes.framing), len(_items(raw.get("framing"), "outlet", "angle")),
            len(kinds), len(_raw_differences(raw)),
            kinds.count("conflict"), kinds.count("figures"), kinds.count("unique"),
        )
        return notes

    def _check(self, raw: dict, outlets: set[str]) -> StoryNotes:
        """Keep notes about real outlets, each difference honest to its kind.

        conflict / figures need two outlets saying different things (a
        figure within 2% is rounding, not a difference); unique needs one
        or two outlets actually reporting the detail. Positions that only
        describe an omission or an inference are dropped either way.
        """
        notes = StoryNotes()

        for item in _items(raw.get("framing"), "outlet", "angle"):
            outlet = _match_outlet(item.get("outlet"), outlets)
            angle = str(item.get("angle") or "").strip()
            if outlet and angle and outlet not in notes.framing:
                notes.framing[outlet] = angle

        candidates: list[Discrepancy] = []
        for item in _raw_differences(raw):
            kind = str(item.get("kind") or "conflict").strip().lower()
            if kind not in KIND_ORDER:
                kind = "conflict"

            positions: dict[str, str] = {}
            for position in _items(item.get("positions"), "outlet", "says"):
                outlet = _match_outlet(position.get("outlet"), outlets)
                says = str(position.get("says") or "").strip()
                if (
                    outlet
                    and says
                    and outlet not in positions
                    and not _OMISSION.search(says)
                ):
                    positions[outlet] = says

            values = list(positions.values())
            if kind == "unique":
                valid = 1 <= len(positions) <= 2
            else:
                valid = (
                    len(positions) >= 2
                    and len({_normalize(v) for v in values}) >= 2
                    and not _is_rounding(values)
                )
            if valid:
                candidates.append(Discrepancy(
                    topic=str(item.get("topic") or "").strip(),
                    positions=positions,
                    kind=kind,
                ))

        candidates.sort(key=lambda d: KIND_ORDER[d.kind])
        unique_kept = 0
        for difference in candidates:
            if difference.kind == "unique":
                if unique_kept >= MAX_UNIQUE:
                    continue
                unique_kept += 1
            notes.discrepancies.append(difference)
            if len(notes.discrepancies) >= MAX_DIFFERENCES:
                break

        return notes

    def _build_prompt(self, cluster: EnrichedCluster, claims: list[Claim]) -> str:
        parts: list[str] = ["## SOURCE ARTICLES\n"]
        for article in cluster.articles:
            parts.append(f"### [{article.entry.outlet}] ({article.source_label})")
            parts.append(f"Title: {article.entry.title}")
            parts.append(
                _truncate_words(article.text, self._config.analysis_max_article_words)
            )
            parts.append("")

        parts.append("## VERIFIED CLAIMS (already settled; for context only)\n")
        for claim in claims:
            parts.append(f'- "{claim.anchor_text}"')
        parts.append("")

        parts.append(f"## OUTLETS\n{', '.join(sorted(cluster.outlets))}\n")
        parts.append(INSTRUCTIONS)
        return "\n".join(parts)
