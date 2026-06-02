"""Sliding-window construction for the translation prompt.

v3 uses a pure-consecutive window: take the previous ``window_size``
parts in story order, skipping any that are in the holdout set, fail to
load, or fall outside :data:`translator.prep.pov.SUPPORTED_POVS`.

Reference ordering places the most recent part *last*, immediately
before the new chapter, so the model attends to it most strongly.

Maika POV side-stories and other special-case entries are filtered out
of the window entirely; the v3 pipeline is not configured to translate
them and using them as in-context examples would dilute the
miyagi-vs-sendai voice signal the prompt depends on.

The window does *not* assemble the prompt itself; it returns an ordered
list of :class:`ReferencePart` objects which :mod:`prompt` then renders.

When the corpus runs out of human-translated references (e.g. when
translating chapters past the last human-translated part), callers can
append AI-translated references via :func:`build_ai_reference`. These
are rendered with a clear warning label so the model uses them for
narrative continuity but does not imitate their style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from translator.config import THRESHOLDS
from translator.prep.corpus import Part, iter_parts, load_part_jp
from translator.prep.holdout import HoldoutPlan
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.scraper.models import POV, SUPPORTED_POVS, TocEntry


@dataclass
class ReferencePart:
    """A single part in the sliding window.

    ``is_ai_translated`` flags references whose EN text was produced by
    the v3 pipeline itself, not by the human reference translator.
    Those references are rendered with a warning header so the LLM
    knows to use them only for plot continuity, not as a style anchor.
    ``ai_source_label`` carries a short provenance string (typically
    the directory name under ``data/output/``) for debug visibility.
    """

    entry: TocEntry
    jp_text: str
    en_text: str
    is_ai_translated: bool = False
    ai_source_label: str | None = None


@dataclass
class Window:
    """The full reference window for a translation call."""

    target_part_id: str
    target_pov: POV
    parts: list[ReferencePart] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _eligible_parts(
    *,
    target_entry: TocEntry,
    lookup: POVLookup,
    holdout: HoldoutPlan | None,
) -> list[Part]:
    """Return parts that may serve as window context for ``target_entry``,
    in part-number order, ascending. Excludes:

    * The target itself
    * Parts after the target (no future leakage)
    * Held-out parts (during eval the holdout members are never used as
      context, even when translating a different held-out member)
    * Parts whose POV isn't in :data:`SUPPORTED_POVS`
    * Parts without an EN translation (untranslated future content)

    The :func:`iter_parts` default already enforces the POV filter, but
    we re-check defensively so the window stays correct even if a
    caller passes a custom lookup with novel POVs.
    """
    target_n = target_entry.part_number or 0
    holdout_ids = set(holdout.part_ids) if holdout else set()
    out: list[Part] = []
    for part in iter_parts(lookup, only_translated=True, parts_only=True):
        if part.id == target_entry.id:
            continue
        if part.id in holdout_ids:
            continue
        if part.entry.pov not in SUPPORTED_POVS:
            continue
        if part.part_number is None or part.part_number >= target_n:
            continue
        out.append(part)
    return out


def build_window(
    target_part_id: str,
    *,
    holdout: HoldoutPlan | None = None,
    lookup: POVLookup | None = None,
) -> Window:
    """Assemble the sliding window for translating ``target_part_id``.

    Pure consecutive: the most recent ``THRESHOLDS.window_size`` eligible
    parts before the target. No retrieval, no per-POV injection — those
    were Escalation 1 paths and are intentionally not scaffolded in v3.
    """
    lookup = lookup or load_pov_lookup()
    target_entry = lookup.get(target_part_id)
    target_pov: POV = target_entry.pov

    eligible = _eligible_parts(target_entry=target_entry, lookup=lookup, holdout=holdout)

    notes: list[str] = []
    quota = THRESHOLDS.window_size
    consecutive = eligible[-quota:] if quota else []

    parts: list[ReferencePart] = []
    for p in consecutive:
        if p.en_text is None:
            continue
        parts.append(ReferencePart(entry=p.entry, jp_text=p.jp_text, en_text=p.en_text))

    if len(parts) < quota:
        notes.append(
            f"window underfilled: {len(parts)}/{quota} parts available "
            f"(early in story or many holdout/untranslated neighbors)"
        )

    return Window(
        target_part_id=target_part_id,
        target_pov=target_pov,
        parts=parts,
        notes=notes,
    )


def build_ai_reference(
    part_id: str,
    en_text_path: Path,
    *,
    lookup: POVLookup | None = None,
    source_label: str | None = None,
) -> ReferencePart:
    """Build a :class:`ReferencePart` for an AI-translated chapter.

    Used to extend the sliding window past the last human-translated
    part. The JP text is loaded from the corpus (must already exist on
    disk under ``data/parallel/``); the EN text is read from
    ``en_text_path`` (typically ``data/output/{part_id}-{suffix}/translation.en.txt``).

    The returned ReferencePart carries ``is_ai_translated=True`` so
    :func:`translator.inference.prompt._format_window` renders it with
    an explicit warning header. ``source_label`` defaults to the parent
    directory name of ``en_text_path`` (e.g. ``part_230-v2-rag-deepseek``)
    so the prompt records which AI run produced the reference.
    """
    lookup = lookup or load_pov_lookup()
    entry = lookup.get(part_id)
    jp_text = load_part_jp(part_id)
    en_text = en_text_path.read_text(encoding="utf-8")
    label = source_label or en_text_path.parent.name
    return ReferencePart(
        entry=entry,
        jp_text=jp_text,
        en_text=en_text,
        is_ai_translated=True,
        ai_source_label=label,
    )


__all__ = ["ReferencePart", "Window", "build_ai_reference", "build_window"]
