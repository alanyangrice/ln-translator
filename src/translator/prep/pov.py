"""POV lookup keyed by part id, and POV detection from JP narrator text.

POV is settled in two ways:

* For parts with an EN counterpart, the avelilium ToC carries an explicit
  ``(Miyagi PoV)`` / ``(Sendai PoV)`` tag and the value is written into
  ``data/metadata/toc.json`` at ``scrape toc`` time.
* For JP-only parts (past the EN translation tail), POV is detected from
  the narrator itself by :func:`detect_pov_from_part_content`, which
  counts character-name occurrences in narration paragraphs only. The
  rule is simple: a first-person narrator names the *other* character,
  not themselves, so whichever surname dominates in narration tells you
  who the narrator is *watching* — and the narrator is the other one.

The :func:`detect_pov_from_part_content` heuristic is exposed here, and
the ``translator prep detect-pov`` CLI command applies it across the
``data/parallel/*.jp.json`` files written by the JP scraper, then
rewrites ``toc.json`` with the resolved POVs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

from translator.config import PATHS
from translator.scraper.models import (
    POV,
    EntryKind,
    PartContent,
    TocEntry,
    is_translation_target,
)


@dataclass(frozen=True)
class POVLookup:
    """Read-only view over the ToC keyed by part id."""

    entries: dict[str, TocEntry]

    def get(self, part_id: str) -> TocEntry:
        if part_id not in self.entries:
            raise KeyError(f"Unknown part id: {part_id}")
        return self.entries[part_id]

    def pov(self, part_id: str) -> POV:
        return self.get(part_id).pov

    def kind(self, part_id: str) -> EntryKind:
        return self.get(part_id).kind

    def iter_by_pov(self, pov: POV) -> Iterator[TocEntry]:
        for entry in self.entries.values():
            if entry.pov == pov:
                yield entry

    def part_ids(self) -> list[str]:
        return list(self.entries.keys())

    def parts_only(self, *, supported_pov_only: bool = True) -> list[TocEntry]:
        """Just the numbered ``part`` entries, in part-number order.

        With ``supported_pov_only=True`` (the default), this is exactly the
        v3 pipeline scope — equivalent to filtering on
        :func:`is_translation_target`. Pass ``False`` to include any
        ``kind == "part"`` entries even with non-supported POVs (only
        useful for corpus audits).
        """
        if supported_pov_only:
            predicate = is_translation_target
        else:
            def predicate(e: TocEntry) -> bool:
                return e.kind == "part" and e.part_number is not None

        return sorted(
            (e for e in self.entries.values() if predicate(e)),
            key=lambda e: e.part_number,  # type: ignore[arg-type, return-value]
        )


def load_pov_lookup() -> POVLookup:
    """Load the ToC and return a POV lookup. Raises if the ToC is missing."""
    if not PATHS.toc_json.exists():
        raise FileNotFoundError(
            f"ToC not found at {PATHS.toc_json}. Run `translator scrape toc` first."
        )
    raw = json.loads(PATHS.toc_json.read_text(encoding="utf-8"))
    entries: dict[str, TocEntry] = {}
    for row in raw:
        entry = TocEntry.model_validate(row)
        entries[entry.id] = entry
    return POVLookup(entries=entries)


# ---------------------------------------------------------------------------
# POV detection from JP narrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class POVDetection:
    pov: POV | None  # None when the heuristic can't decide
    sendai_count: int  # mentions of 仙台 in narration
    miyagi_count: int  # mentions of 宮城 in narration
    confident: bool


def detect_pov_from_part_content(content: PartContent, *, min_total: int = 3) -> POVDetection:
    """Infer narrator POV from a JP part's narration paragraphs.

    Strategy: count 仙台 / 宮城 occurrences in narration paragraphs only;
    whoever is named more is the *non-narrator*, so the POV is the
    other character.

    ``min_total`` guards against pathological cases (e.g. a chapter
    where neither name appears in narration). Below that floor we
    return ``pov=None`` and the caller can keep the placeholder or
    flag for review.
    """
    sendai = 0
    miyagi = 0
    for para in content.paragraphs:
        if para.kind != "narration":
            continue
        sendai += para.text.count("仙台")
        miyagi += para.text.count("宮城")

    total = sendai + miyagi
    if total < min_total:
        return POVDetection(pov=None, sendai_count=sendai, miyagi_count=miyagi, confident=False)
    if sendai > miyagi:
        return POVDetection(pov="miyagi", sendai_count=sendai, miyagi_count=miyagi,
                            confident=sendai > 2 * miyagi or sendai - miyagi >= 3)
    if miyagi > sendai:
        return POVDetection(pov="sendai", sendai_count=sendai, miyagi_count=miyagi,
                            confident=miyagi > 2 * sendai or miyagi - sendai >= 3)
    return POVDetection(pov=None, sendai_count=sendai, miyagi_count=miyagi, confident=False)


def detect_pov_from_disk(part_id: str) -> POVDetection | None:
    """Run :func:`detect_pov_from_part_content` against the JP JSON on disk."""
    path = PATHS.parallel / f"{part_id}.jp.json"
    if not path.exists():
        return None
    content = PartContent.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return detect_pov_from_part_content(content)
