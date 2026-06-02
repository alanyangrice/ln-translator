"""Manifest of "blessed" AI-translated chapters for the sliding window.

Once the corpus's human-translated range ends (part_229 in this run),
each new chapter we translate has to lean on a mix of:

1. The 10 most-recent **human** reference translations (parts 220-229) —
   the style anchor.
2. Up to ``THRESHOLDS.ai_reference_window_size`` most-recent **AI**
   reference translations (parts past 229 that we've already produced
   and curated) — the plot-continuity anchor.

This module is the small registry that says "these specific AI runs
are the canonical ones to reference." Without it the
``translator translate`` command would have no way to know *which*
draft to grab from ``data/output/{part_id}-*/`` since there can be
several (different models, ablations, suffixes).

Design
------
* Stored at :attr:`PATHS.ai_references_manifest`
  (``knowledge-vault/ai-references.json``) so it sits beside the rest
  of the curated knowledge.
* JSON, hand-editable. Promotion via
  ``translator ai-ref promote --part part_NNN --suffix v2-rag-deepseek``
  is the normal path; manual edits are fine too.
* Entries record ``en_text`` (path to the translation), ``model``,
  ``source_suffix`` (directory name under ``data/output/``), and a
  ``promoted_at`` timestamp for audit.
* :func:`find_recent_for_target` is the consumer used by
  :mod:`translator.inference.translate`: given a target part id, it
  returns the *N* most recent AI references whose part number is
  strictly less than the target's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from translator.config import PATHS

_PART_NUMBER_RE = re.compile(r"part_(\d+)")


def _part_number(part_id: str) -> int:
    """Parse the numeric suffix of a part id (``part_230 -> 230``)."""
    m = _PART_NUMBER_RE.fullmatch(part_id)
    if not m:
        raise ValueError(f"invalid part id: {part_id!r}")
    return int(m.group(1))


@dataclass
class AIReferenceEntry:
    """One blessed AI translation."""

    part_id: str
    en_text: Path
    model: str = ""
    source_suffix: str = ""
    promoted_at: str = ""

    def to_record(self) -> dict:
        return {
            "en_text": str(self.en_text),
            "model": self.model,
            "source_suffix": self.source_suffix,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def from_record(cls, part_id: str, rec: dict) -> "AIReferenceEntry":
        return cls(
            part_id=part_id,
            en_text=Path(rec["en_text"]),
            model=rec.get("model", ""),
            source_suffix=rec.get("source_suffix", ""),
            promoted_at=rec.get("promoted_at", ""),
        )


@dataclass
class AIReferenceManifest:
    """Whole manifest. ``entries`` is keyed by part id."""

    entries: dict[str, AIReferenceEntry] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def add(self, entry: AIReferenceEntry) -> None:
        self.entries[entry.part_id] = entry

    def remove(self, part_id: str) -> bool:
        return self.entries.pop(part_id, None) is not None

    def sorted_by_part(self, *, ascending: bool = True) -> list[AIReferenceEntry]:
        return sorted(
            self.entries.values(),
            key=lambda e: _part_number(e.part_id),
            reverse=not ascending,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "entries": {
                    pid: self.entries[pid].to_record()
                    for pid in sorted(
                        self.entries.keys(), key=_part_number
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )


def load_manifest(path: Path | None = None) -> AIReferenceManifest:
    """Load the manifest. Returns an empty manifest if the file is missing."""
    path = path or PATHS.ai_references_manifest
    if not path.exists():
        return AIReferenceManifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, AIReferenceEntry] = {}
    for pid, rec in data.get("entries", {}).items():
        entries[pid] = AIReferenceEntry.from_record(pid, rec)
    return AIReferenceManifest(entries=entries)


def save_manifest(
    manifest: AIReferenceManifest, path: Path | None = None
) -> None:
    """Persist the manifest as pretty JSON, creating parents as needed."""
    path = path or PATHS.ai_references_manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def find_recent_for_target(
    target_part_id: str,
    *,
    limit: int,
    manifest: AIReferenceManifest | None = None,
) -> list[AIReferenceEntry]:
    """Return up to ``limit`` AI references with part_number < target.

    Sorted **ascending** by part number so the caller can append them
    to the window in story order (most recent last). Entries whose
    ``en_text`` path no longer exists on disk are skipped with a
    ``FileNotFoundError`` — manifest entries should always point to
    extant files, and a missing one almost certainly means the user
    deleted the underlying draft directory by mistake.
    """
    if limit <= 0:
        return []
    manifest = manifest if manifest is not None else load_manifest()
    target_n = _part_number(target_part_id)
    candidates = [
        e
        for e in manifest.entries.values()
        if _part_number(e.part_id) < target_n
    ]
    candidates.sort(key=lambda e: _part_number(e.part_id), reverse=True)
    candidates = candidates[:limit]
    candidates.sort(key=lambda e: _part_number(e.part_id))

    for entry in candidates:
        if not entry.en_text.exists():
            raise FileNotFoundError(
                f"AI reference for {entry.part_id} not found on disk: "
                f"{entry.en_text}. Re-promote with "
                f"`translator ai-ref promote --part {entry.part_id} "
                f"--from <path>` or remove the stale manifest entry."
            )
    return candidates


def promote(
    part_id: str,
    *,
    en_text: Path | None = None,
    source_suffix: str | None = None,
    model: str = "",
    manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> AIReferenceEntry:
    """Add or replace a manifest entry.

    Exactly one of ``en_text`` or ``source_suffix`` must be supplied.
    When ``source_suffix`` is given, the path is resolved as
    ``{output_root}/{part_id}-{source_suffix}/translation.en.txt``.
    """
    if (en_text is None) == (source_suffix is None):
        raise ValueError(
            "promote() requires exactly one of en_text or source_suffix"
        )
    output_root = output_root or PATHS.output
    if source_suffix is not None:
        en_text = output_root / f"{part_id}-{source_suffix}" / "translation.en.txt"
        source_suffix_value = source_suffix
    else:
        assert en_text is not None
        source_suffix_value = en_text.parent.name.replace(f"{part_id}-", "", 1)
        if source_suffix_value == en_text.parent.name:
            # Parent dir name didn't start with "{part_id}-"; record verbatim.
            source_suffix_value = en_text.parent.name

    if not en_text.exists():
        raise FileNotFoundError(
            f"refusing to promote — file not found: {en_text}"
        )
    if not model:
        meta_path = en_text.parent / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model = meta.get("model", "")
            except (json.JSONDecodeError, OSError):
                model = ""

    manifest = load_manifest(manifest_path)
    entry = AIReferenceEntry(
        part_id=part_id,
        en_text=en_text,
        model=model,
        source_suffix=source_suffix_value,
        promoted_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    manifest.add(entry)
    save_manifest(manifest, manifest_path)
    return entry


__all__ = [
    "AIReferenceEntry",
    "AIReferenceManifest",
    "find_recent_for_target",
    "load_manifest",
    "promote",
    "save_manifest",
]
