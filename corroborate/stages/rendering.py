from __future__ import annotations

from datetime import datetime

from corroborate.models import StoryBrief


def _generation_heading(now: datetime | None = None) -> str:
    """Build the document title from the clock, not the LLM."""
    now = now or datetime.now()
    clock = now.strftime("%I:%M %p").lstrip("0")
    return f"# Daily Brief: {clock}, {now.strftime('%B')} {now.day}, {now.year}"


def default_output_filename(now: datetime | None = None) -> str:
    """Name a saved brief by its date, e.g. OCT_03_2026.md."""
    now = now or datetime.now()
    return f"{now.strftime('%b').upper()}_{now.day:02d}_{now.year}.md"


class RenderingStage:
    """Assembles the finished document. Pure code: no model rewrites it."""

    def render(
        self,
        overview: str | None,
        stories: list[StoryBrief],
        digest: StoryBrief | None,
        now: datetime | None = None,
    ) -> str:
        parts = [_generation_heading(now)]
        if overview:
            parts.append(overview.strip())
        parts.extend(story.brief_text.strip() for story in stories)
        if digest:
            parts.append(digest.brief_text.strip())

        sources = {outlet for story in stories for outlet in story.outlets}
        if digest:
            sources.update(digest.outlets)
        parts.append(f"Sources: {', '.join(sorted(sources))}.")

        return "\n\n".join(parts) + "\n"
