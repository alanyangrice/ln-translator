"""Pydantic models for the ToC + per-part parallel artifact format.

The per-paragraph schema preserves ``kind`` markers (narration / dialogue /
blank) so:

* :mod:`translator.scraper.persist` can render the ``.txt`` form with empty
  lines for scene breaks (the prompt's only scene-boundary signal).
* :mod:`translator.scraper.align` can flag JP/EN paragraph-count skew at
  scrape time.
* Future paragraph-level validators can attribute by structure without
  re-parsing the raw HTML.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ParagraphKind = Literal["narration", "dialogue", "blank", "heading"]
POV = Literal["miyagi", "sendai", "maika", "both"]
EntryKind = Literal["part", "interlude", "extra", "side_story_maika", "bookwalker", "special"]

# POVs the v3 pipeline is configured to handle. Maika POV (only in
# side-story entries) and "both" POV (alternating-narrator chapters)
# would each need bespoke prompt handling and are intentionally out of
# scope.
SUPPORTED_POVS: frozenset[POV] = frozenset({"miyagi", "sendai"})


class TocEntry(BaseModel):
    """One row in the unified JP/EN table of contents."""

    id: str  # stable identifier, e.g. "part_001", "interlude_v1", "side_story_maika_v2"
    kind: EntryKind
    part_number: int | None = None
    chapter_number: int | None = None
    chapter_subnumber: float | None = None  # e.g. 5.5 for interludes
    volume_number: int
    pov: POV
    title_en: str
    url_en: str | None = None
    kakuyomu_episode_id: str | None = None
    kakuyomu_chapter_id: str | None = None
    url_jp: str | None = None
    chapter_title_en: str | None = None
    chapter_title_jp: str | None = None
    read_after_part: int | None = None
    mapping_confidence: Literal["auto", "manual", "needs_review"] = "auto"
    notes: str | None = None


class Paragraph(BaseModel):
    index: int
    kind: ParagraphKind
    text: str = ""


class PartContent(BaseModel):
    """One side (JP or EN) of a single part / interlude / extra / SS."""

    id: str
    side: Literal["jp", "en"]
    source_url: str
    title_native: str | None = None
    paragraphs: list[Paragraph] = Field(default_factory=list)

    @property
    def non_blank_count(self) -> int:
        return sum(1 for p in self.paragraphs if p.kind != "blank")


def is_translation_target(entry: TocEntry) -> bool:
    """Single source of truth for "is this entry in v3 pipeline scope?".

    Returns True only for the numbered main-story chapters (POV miyagi
    or sendai). Interludes, extras, side stories (Maika), bonus content,
    and alternating-POV chapters all return False and are skipped at
    every layer: scrape, verify, calibrate, holdout, sliding window,
    and translate.
    """
    return (
        entry.kind == "part"
        and entry.part_number is not None
        and entry.pov in SUPPORTED_POVS
    )
