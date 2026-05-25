"""Per-chapter deviation note writers.

The comparison step (``eval.deviations``) emits one Markdown note per
chapter under ``deviations/round-NN/part-XXX-deviations.md``. The
clustering step reads these notes, groups them into candidate rules,
and writes the resulting rule notes via :mod:`translator.vault.rules`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from translator.config import PATHS, VAULT
from translator.vault.notes import Note, write_note

DeviationCategory = Literal[
    "tense",
    "voice/register",
    "attribution",
    "glossary",
    "sentence-structure",
    "omission",
    "addition",
    "idiom",
    "pronoun",
    "formatting",
    "translationese",
    "style-rhythm",
    "style-profile",
]
Severity = Literal["minor", "major"]


@dataclass
class Deviation:
    category: DeviationCategory
    severity: Severity
    pov_specific: bool
    jp_source: str
    llm_rendering: str
    reference_rendering: str
    notes: str = ""
    violates_rule_id: str = ""  # empty = no active rule violated; otherwise e.g. "rule-000-06"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "pov_specific": self.pov_specific,
            "jp_source": self.jp_source,
            "llm_rendering": self.llm_rendering,
            "reference_rendering": self.reference_rendering,
            "notes": self.notes,
            "violates_rule_id": self.violates_rule_id,
        }


@dataclass
class DeviationNote:
    """The full deviation report for one chapter in one round."""

    part_id: str
    round_number: int
    pov: str
    deviations: list[Deviation] = field(default_factory=list)
    chapter_type: str | None = None  # e.g. "dialogue_heavy", filled by eval if classified

    def filename(self) -> str:
        return f"{self.part_id.replace('_', '-')}-deviations.md"

    def round_dir_name(self) -> str:
        return f"round-{self.round_number:02d}"

    def deviation_id(self, idx: int) -> str:
        """Return the canonical ID for the i-th deviation (1-indexed).

        Format: ``part-{chapter:03d}-r{round:02d}-d{idx:02d}`` — e.g.
        ``part-012-r01-d01``. The clustering step parses chapter numbers
        out of this string to name candidate rules, so the format must
        stay stable. Two-digit indices keep IDs sortable lexicographically.
        """
        chapter = self.part_id.removeprefix("part_")
        return f"part-{chapter}-r{self.round_number:02d}-d{idx:02d}"

    def to_note(self) -> Note:
        meta: dict[str, Any] = {
            "part_id": self.part_id,
            "round": self.round_number,
            "pov": self.pov,
            "deviation_count": len(self.deviations),
        }
        if self.chapter_type:
            meta["chapter_type"] = self.chapter_type

        lines = [f"# Deviations: {self.part_id}, Round {self.round_number:02d}", ""]
        if not self.deviations:
            lines.append("_No deviations flagged._")
        for i, d in enumerate(self.deviations, 1):
            dev_id = self.deviation_id(i)
            lines.extend(
                [
                    f"## {dev_id}",
                    "",
                    f"- **ID:** `{dev_id}`",
                    f"- **Category:** {d.category}",
                    f"- **Severity:** {d.severity}",
                    f"- **POV-specific:** {str(d.pov_specific).lower()}",
                    f"- **JP source:** {d.jp_source}",
                    f"- **LLM rendering:** {d.llm_rendering}",
                    f"- **Reference rendering:** {d.reference_rendering}",
                ]
            )
            if d.violates_rule_id:
                lines.append(f"- **Violates rule:** [[{d.violates_rule_id}]]")
            if d.notes:
                lines.append(f"- **Notes:** {d.notes}")
            lines.append("")
        return Note(meta=meta, body="\n".join(lines))


def write_deviation_note(note: DeviationNote, root: Path | None = None) -> Path:
    """Persist a deviation note to ``deviations/round-NN/part-XXX-deviations.md``."""
    root = root or PATHS.knowledge_vault
    target = root / VAULT.deviations / note.round_dir_name() / note.filename()
    write_note(target, note.to_note())
    return target
