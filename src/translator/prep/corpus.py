"""Loaders for the parallel corpus on disk.

Each part has up to four files in ``data/parallel/``:

* ``part_NNN.jp.txt`` — the JP raw text the prompt feeds to the model.
* ``part_NNN.en.txt`` — the EN raw text from the human translator.
* ``part_NNN.jp.json`` — the structured paragraph version (used by validators).
* ``part_NNN.en.json`` — the structured paragraph version.

Untranslated parts (Phase 9 of the v3 implementation order) only have
``.jp.*`` files; ``en_text`` will be ``None`` for those.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

from translator.config import PATHS
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.scraper.models import PartContent, TocEntry


@dataclass
class Part:
    """A single part of the corpus, JP + (optional) EN.

    ``jp_text`` and ``en_text`` are the raw ``.txt`` strings the
    translator feeds to the model. ``jp_paragraphs`` and ``en_paragraphs``
    are the structured paragraph versions used by validators that need
    paragraph-level granularity (dialogue parity, etc.).
    """

    entry: TocEntry
    jp_text: str
    en_text: str | None = None
    jp_paragraphs: PartContent | None = None
    en_paragraphs: PartContent | None = None

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def part_number(self) -> int | None:
        return self.entry.part_number

    @property
    def has_translation(self) -> bool:
        return self.en_text is not None


def _read_text(path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _read_part_content(path) -> PartContent | None:
    if not path.exists():
        return None
    try:
        return PartContent.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def load_part(part_id: str, lookup: POVLookup | None = None) -> Part:
    """Load all available files for a part. Raises if no JP source is on disk."""
    lookup = lookup or load_pov_lookup()
    entry = lookup.get(part_id)
    jp_text = _read_text(PATHS.parallel / f"{part_id}.jp.txt")
    if jp_text is None:
        raise FileNotFoundError(
            f"JP source missing for {part_id}; run `translator scrape jp --only {part_id}`."
        )
    return Part(
        entry=entry,
        jp_text=jp_text,
        en_text=_read_text(PATHS.parallel / f"{part_id}.en.txt"),
        jp_paragraphs=_read_part_content(PATHS.parallel / f"{part_id}.jp.json"),
        en_paragraphs=_read_part_content(PATHS.parallel / f"{part_id}.en.json"),
    )


def load_part_jp(part_id: str) -> str:
    text = _read_text(PATHS.parallel / f"{part_id}.jp.txt")
    if text is None:
        raise FileNotFoundError(f"Missing JP text for {part_id}")
    return text


def load_part_en(part_id: str) -> str | None:
    return _read_text(PATHS.parallel / f"{part_id}.en.txt")


def iter_parts(
    lookup: POVLookup | None = None,
    *,
    only_translated: bool = False,
    parts_only: bool = True,
    supported_pov_only: bool = True,
) -> Iterator[Part]:
    """Iterate over corpus parts in canonical order.

    ``parts_only=True`` (default) skips interludes/extras/side-stories.
    ``supported_pov_only=True`` (default) further filters to POVs the
    v3 pipeline is set up to handle (see :data:`SUPPORTED_POVS`); pass
    ``False`` only when you explicitly want a corpus-wide audit view.
    ``only_translated=True`` skips entries whose ``.en.txt`` is missing.
    Entries whose JP source isn't on disk yet are silently skipped.
    """
    from translator.scraper.models import SUPPORTED_POVS

    lookup = lookup or load_pov_lookup()
    if parts_only:
        entries = lookup.parts_only(supported_pov_only=supported_pov_only)
    else:
        all_entries = sorted(
            lookup.entries.values(), key=lambda e: (e.part_number or 1_000_000, e.id)
        )
        entries = (
            [e for e in all_entries if e.pov in SUPPORTED_POVS]
            if supported_pov_only
            else all_entries
        )
    for entry in entries:
        try:
            part = load_part(entry.id, lookup)
        except FileNotFoundError:
            continue
        if only_translated and not part.has_translation:
            continue
        yield part
