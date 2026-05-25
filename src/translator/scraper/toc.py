"""Build the unified JP↔EN table of contents.

Strategy:

1.  Parse the avelilium ToC page into typed entries (one per translated unit).
    POV is read from the explicit "(Xxx PoV)" tag on each chapter heading line.
    Volume number is derived from the "This is where volume N ends" separators.

2.  Pull the Kakuyomu work-level chapter tree (ordered chapters → episodes).

3.  Map each EN entry to a Kakuyomu episode by walking both sides in lockstep
    on the main story chapters. Side stories / interludes / extras are mapped
    by matching the chapter title contents (kakuyomu chapter titles for
    interludes / extras carry tell-tale prefixes like "幕間" or "番外編").

4.  Anything we cannot confidently map is marked ``mapping_confidence='needs_review'``
    so a human can audit ``toc.json`` before running the JP scraper.

The expected steady state: 229 ``part`` entries + 7-8 interludes + 7-8 extras
+ 4 Maika side stories + 1 Maika interlude + 1 Bookwalker SS = ~250 entries.
"""

from __future__ import annotations

import re

from translator.config import PATHS
from translator.scraper.avelilium import (
    _POV_TAG_RE,
    TocLink,
    _chapter_number_from_block,
    _chapter_title_from_block,
    fetch_toc_html,
    parse_toc_links,
)
from translator.scraper.kakuyomu import KakuyomuChapter, episode_url, fetch_work_chapters
from translator.scraper.models import POV, EntryKind, TocEntry

# ---------------------------------------------------------------------------
# EN-side typed entry extraction
# ---------------------------------------------------------------------------


_VOL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}


def _volume_from_block(block: str) -> int | None:
    m = re.search(r"Volume\s+([A-Za-z]+)", block, re.IGNORECASE)
    if not m:
        return None
    return _VOL_WORDS.get(m.group(1).lower())


_READ_AFTER_RE = re.compile(r"read after part\s+(\d+)", re.IGNORECASE)


def _classify_link(link: TocLink) -> EntryKind:
    """Classify a single ToC link by inspecting its block text + label.

    Order of checks matters: ``special`` and ``bookwalker`` blocks may also
    contain "Part" / "Side Story" keywords, so they are matched first.
    """
    label = link.label.lower()
    block_lower = link.chapter_block.lower()

    if "april fool" in block_lower or "special chapter" in block_lower:
        return "special"
    if "bookwalker" in block_lower or "bookwalker" in label:
        return "bookwalker"
    if "maika side story" in block_lower or "maika ss" in block_lower:
        return "side_story_maika"
    if "interlude" in block_lower:
        return "interlude"
    if "extra" in block_lower and "extra" in label:
        return "extra"
    if label.startswith("part"):
        return "part"
    if "side story" in block_lower:
        return "side_story_maika"
    return "part"


def _entry_id_for(kind: EntryKind, part_number: int | None, volume: int, vol_index: int) -> str:
    if kind == "part" and part_number is not None:
        return f"part_{part_number:03d}"
    if kind == "interlude":
        return f"interlude_v{volume}"
    if kind == "extra":
        return f"extra_v{volume}"
    if kind == "bookwalker":
        return f"bookwalker_v{volume}"
    if kind == "side_story_maika":
        return f"maika_ss_v{volume}"
    if kind == "special":
        return f"special_{volume:02d}"
    return f"unknown_v{volume}_{vol_index:02d}"


def assign_volumes(links: list[TocLink], volume_breaks: list[int]) -> list[int]:
    """Return parallel list of volume numbers per link."""
    vols: list[int] = []
    for link in links:
        # Volume number = number of volume-break separators strictly less than this block index, + 1.
        vol = 1 + sum(1 for vb in volume_breaks if vb < link.block_index)
        vols.append(vol)
    return vols


def _pov_for_entry(kind: EntryKind, block: str, label: str) -> POV:
    """Determine POV. Two key special cases:

    * The Vol. 6 Extra block reads "... (Sendai PoV) ... Volume Six Extra ... (Both PoVs) [Extra]".
      The chapter heading PoV is Sendai, but the Extra's own PoV is "Both".
    * Maika Side Story labels don't carry a PoV tag in the heading; default to ``maika``.
    """
    if kind == "side_story_maika":
        return "maika"

    # Find all POV tags in the block, then choose the one closest to (and preceding) the link label.
    tag_iter = list(_POV_TAG_RE.finditer(block))
    if not tag_iter:
        # Vol-level fallback by entry kind defaults.
        if kind == "interlude":
            return "sendai"
        return "miyagi"

    # Anchor on the LAST occurrence of the label text in the block so we pick
    # up the right POV tag for the Vol. 6 Extra case where the block contains
    # both "(Sendai PoV)" (the parent chapter) and "(Both PoVs)" (the Extra's
    # own tag). The label "Extra" also appears in "Volume Six Extra" earlier
    # in the block, so rfind is the safer anchor.
    label_idx = block.rfind(label)
    if label_idx >= 0:
        preceding = [m for m in tag_iter if m.start() < label_idx]
        chosen = preceding[-1] if preceding else tag_iter[0]
    else:
        chosen = tag_iter[0]

    tag = chosen.group("pov").lower()
    if tag in ("miyagi", "sendai", "maika"):
        return tag  # type: ignore[return-value]
    return "both"


def extract_en_entries(html: str) -> list[TocEntry]:
    links, vol_breaks = parse_toc_links(html)
    vols = assign_volumes(links, vol_breaks)

    entries: list[TocEntry] = []
    vol_seen_counter: dict[int, int] = {}

    for link, vol in zip(links, vols, strict=False):
        block = link.chapter_block
        chapter_num = _chapter_number_from_block(block)
        chapter_title = _chapter_title_from_block(block)
        kind = _classify_link(link)
        pov = _pov_for_entry(kind, block, link.label)

        part_match = re.match(r"^part\s+(\d+)", link.label, re.IGNORECASE)
        part_number = int(part_match.group(1)) if part_match else None
        # Special chapter "Part 1" must NOT carry a numbered part id.
        if kind == "special":
            part_number = None

        # Volume override: side stories carry "Vol. N" in their block.
        if kind == "side_story_maika":
            m = re.search(r"vol\.\s*(\d+)", block, re.IGNORECASE)
            if m:
                vol = int(m.group(1))

        read_after = None
        ra = _READ_AFTER_RE.search(block)
        if ra:
            read_after = int(ra.group(1))

        vol_seen_counter[vol] = vol_seen_counter.get(vol, 0) + 1
        entry_id = _entry_id_for(kind, part_number, vol, vol_seen_counter[vol])

        sub = None
        if kind == "interlude" and chapter_num is not None:
            sub = chapter_num + 0.5

        # LN-exclusive entries have no Kakuyomu web-novel source.
        ln_only = kind in ("interlude", "extra", "side_story_maika", "bookwalker", "special")
        confidence = "manual" if ln_only else "auto"
        notes = "LN-exclusive (no web-novel source on Kakuyomu)" if ln_only else None

        entries.append(
            TocEntry(
                id=entry_id,
                kind=kind,
                part_number=part_number,
                chapter_number=chapter_num,
                chapter_subnumber=sub,
                volume_number=vol,
                pov=pov,
                title_en=link.label,
                url_en=link.href,
                chapter_title_en=chapter_title,
                read_after_part=read_after,
                mapping_confidence=confidence,
                notes=notes,
            )
        )

    return entries


# ---------------------------------------------------------------------------
# JP-side mapping
# ---------------------------------------------------------------------------

# Sentinel value usable in the manual overrides file:
#   "<entry_id>": "LN_ONLY"     →  this entry has no Kakuyomu source (light novel only)
#   "<entry_id>": "<episode_id>" →  force-bind to a specific Kakuyomu episode id
_LN_ONLY_SENTINEL = "LN_ONLY"


def _load_manual_overrides() -> dict[str, str]:
    """Optional ``data/metadata/kakuyomu_overrides.json`` lets a human pin the
    JP mapping for specific entries when the auto-matcher gets it wrong."""
    path = PATHS.metadata / "kakuyomu_overrides.json"
    if not path.exists():
        return {}
    try:
        import json as _json
        return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _apply_kakuyomu_chapter(
    entry: TocEntry,
    chapter: KakuyomuChapter,
    episode_index: int,
) -> None:
    if episode_index >= len(chapter.episode_ids):
        entry.mapping_confidence = "needs_review"
        entry.notes = (
            f"kakuyomu chapter {chapter.title!r} has only {len(chapter.episode_ids)} episodes "
            f"but EN expected episode #{episode_index + 1}"
        )
        return
    ep_id = chapter.episode_ids[episode_index]
    entry.kakuyomu_episode_id = ep_id
    entry.kakuyomu_chapter_id = chapter.id
    entry.chapter_title_jp = chapter.title
    entry.url_jp = episode_url(ep_id)


def _flatten_episodes(
    chapters: list[KakuyomuChapter],
) -> list[tuple[str, KakuyomuChapter, int]]:
    """Return flat list of (episode_id, chapter, episode_index_in_chapter) in story order."""
    out: list[tuple[str, KakuyomuChapter, int]] = []
    for ch in chapters:
        for idx, eid in enumerate(ch.episode_ids):
            out.append((eid, ch, idx))
    return out


def map_to_kakuyomu(
    entries: list[TocEntry],
    chapters: list[KakuyomuChapter],
) -> list[TocEntry]:
    """Map each EN ``part`` entry to a Kakuyomu episode by sequential index.

    Kakuyomu episodes for this work are titled exactly ``第1話`` … ``第417話``
    in story order, so ``part_N`` corresponds 1:1 to KU episode ``#N``. This
    is the simplest possible mapping and matches the source-of-truth: the
    human translator's avelilium parts are numbered against the same
    underlying episode sequence.

    LN-exclusive entries (``kind`` in interlude/extra/side_story_maika/
    bookwalker/special) are left unmapped — they don't exist on the web
    novel — unless a manual override pins them.
    """
    out: list[TocEntry] = list(entries)
    overrides = _load_manual_overrides()
    flat = _flatten_episodes(chapters)

    for i, e in enumerate(out):
        if e.kind != "part" or e.part_number is None:
            continue
        idx = e.part_number - 1
        if idx < 0 or idx >= len(flat):
            out[i].mapping_confidence = "needs_review"
            out[i].notes = (
                f"part_number {e.part_number} out of Kakuyomu episode range "
                f"(have {len(flat)} episodes)"
            )
            continue
        eid, kch, ep_idx = flat[idx]
        out[i].kakuyomu_episode_id = eid
        out[i].kakuyomu_chapter_id = kch.id
        out[i].chapter_title_jp = kch.title
        out[i].url_jp = episode_url(eid)
        out[i].mapping_confidence = "auto"

    # Manual overrides win (rarely needed now that mapping is deterministic).
    for i, e in enumerate(out):
        override = overrides.get(e.id)
        if not override:
            continue
        if override == _LN_ONLY_SENTINEL:
            out[i].kakuyomu_episode_id = None
            out[i].kakuyomu_chapter_id = None
            out[i].url_jp = None
            out[i].mapping_confidence = "manual"
            out[i].notes = "manual override: LN-exclusive"
            continue
        owning_chapter: KakuyomuChapter | None = None
        ep_index = 0
        for c in chapters:
            if override in c.episode_ids:
                owning_chapter = c
                ep_index = c.episode_ids.index(override)
                break
        if owning_chapter is None:
            out[i].mapping_confidence = "needs_review"
            out[i].notes = f"manual override episode id {override!r} not found on Kakuyomu"
            continue
        _apply_kakuyomu_chapter(out[i], owning_chapter, ep_index)
        out[i].mapping_confidence = "manual"
        out[i].notes = "manual override"

    return out


def add_jp_only_parts(
    entries: list[TocEntry],
    chapters: list[KakuyomuChapter],
) -> list[TocEntry]:
    """Append JP-only ``part_N`` entries for Kakuyomu episodes past the EN tail.

    The EN translation tracks the JP web novel chronologically but with a
    delay; everything beyond the last EN-mapped part exists on Kakuyomu
    only. We surface those as proper ``kind="part"`` entries with no
    ``url_en`` so the JP scraper picks them up but the EN scraper skips
    them.

    POV is set to a placeholder (``miyagi``) with ``mapping_confidence
    == "needs_review"``; ``prep detect-pov`` (run after the JP scrape)
    rewrites it from the actual narrator.
    """
    flat = _flatten_episodes(chapters)
    existing_part_numbers = {e.part_number for e in entries if e.kind == "part"}

    # Place JP-only parts at the highest existing volume + 1 by default.
    max_vol = max((e.volume_number for e in entries), default=1)
    next_vol = max_vol + 1

    appended: list[TocEntry] = []
    for idx, (eid, kch, _ep_idx) in enumerate(flat):
        part_n = idx + 1
        if part_n in existing_part_numbers:
            continue
        appended.append(
            TocEntry(
                id=f"part_{part_n:03d}",
                kind="part",
                part_number=part_n,
                chapter_number=None,
                volume_number=next_vol,
                pov="miyagi",  # placeholder; refined by prep detect-pov
                title_en=f"Part {part_n}",
                url_en=None,
                kakuyomu_episode_id=eid,
                kakuyomu_chapter_id=kch.id,
                url_jp=episode_url(eid),
                chapter_title_jp=kch.title,
                mapping_confidence="needs_review",
                notes="JP-only (past EN translation tail); POV needs detection",
            )
        )

    return list(entries) + appended


# ---------------------------------------------------------------------------
# Public entrypoint used by the CLI.
# ---------------------------------------------------------------------------


def build_toc(*, refresh: bool = False) -> list[TocEntry]:
    en_html = fetch_toc_html(refresh=refresh)
    entries = extract_en_entries(en_html)
    chapters = fetch_work_chapters(refresh=refresh)
    mapped = map_to_kakuyomu(entries, chapters)
    return add_jp_only_parts(mapped, chapters)
