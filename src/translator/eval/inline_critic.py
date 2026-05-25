"""Inference-time critic that flags translationese in a draft.

Sister module to :mod:`translator.eval.deviations`. The deviation
auditor compares a draft against the human reference and writes notes
into the vault for the offline self-improvement loop. The critic here
runs *during translation*: no reference is available, the goal is to
catch calques and voice drift before the draft is persisted, and the
output drives a revision pass rather than rule promotion.

Reuse vs. fork relative to the deviation auditor:

* Reused — rule/glossary/style-profile loaders, JSON-schema parsing
  utilities, provider call shape, category/severity enums.
* Forked — the prompt is reference-free; the per-flag payload uses
  ``span``/``suggested_rewrite`` instead of ``llm_rendering``/
  ``reference_rendering``; persistence lives next to the translation
  artifacts in ``data/output/`` rather than in the vault.

The categories enum mirrors :data:`translator.vault.deviations.DeviationCategory`
verbatim so a single mental model covers both pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Literal

from translator.config import MODELS, PATHS, REASONING, VAULT
from translator.glossary import format_for_prompt as format_glossary_for_prompt
from translator.glossary import load_glossary
from translator.inference.provider import complete
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.style import format_style_profile_for_prompt
from translator.vault import format_rules_for_prompt, load_active_rules
from translator.vault.deviations import DeviationCategory, Severity
from translator.vault.templates import CRITIQUE_TEMPLATE


@dataclass
class CritiqueFlag:
    """A single inline-critic finding.

    ``span`` is the verbatim substring of the draft to be replaced;
    ``suggested_rewrite`` is the critic's proposed substitute. Both
    are required so the revision pass can locate and apply the fix
    without re-deriving it.
    """

    category: DeviationCategory
    severity: Severity
    pov_specific: bool
    jp_source: str
    span: str
    suggested_rewrite: str
    notes: str = ""
    violates_rule_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "pov_specific": self.pov_specific,
            "jp_source": self.jp_source,
            "span": self.span,
            "suggested_rewrite": self.suggested_rewrite,
            "notes": self.notes,
            "violates_rule_id": self.violates_rule_id,
        }


@dataclass
class CritiqueResult:
    """The full critique payload for one draft."""

    part_id: str
    pov: str
    model: str
    flags: list[CritiqueFlag] = field(default_factory=list)
    raw: str = ""

    @property
    def major_flags(self) -> list[CritiqueFlag]:
        return [f for f in self.flags if f.severity == "major"]

    @property
    def minor_flags(self) -> list[CritiqueFlag]:
        return [f for f in self.flags if f.severity == "minor"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "pov": self.pov,
            "model": self.model,
            "flag_count": len(self.flags),
            "major_count": len(self.major_flags),
            "minor_count": len(self.minor_flags),
            "flags": [f.to_dict() for f in self.flags],
        }

    def write(self, path: Path | None = None) -> Path:
        """Persist the critique as ``data/output/<part>/critique.json``."""
        target = path or (PATHS.output / self.part_id / "critique.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


CRITIQUE_JSON_SCHEMA: dict = {
    "name": "critique",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
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
                            ],
                        },
                        "severity": {"type": "string", "enum": ["minor", "major"]},
                        "pov_specific": {"type": "boolean"},
                        "jp_source": {"type": "string"},
                        "span": {"type": "string"},
                        "suggested_rewrite": {"type": "string"},
                        "notes": {"type": "string"},
                        "violates_rule_id": {"type": "string"},
                    },
                    "required": [
                        "category",
                        "severity",
                        "pov_specific",
                        "jp_source",
                        "span",
                        "suggested_rewrite",
                        "notes",
                        "violates_rule_id",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["flags"],
        "additionalProperties": False,
    },
}


def _load_critique_template() -> str:
    vault_path = PATHS.knowledge_vault / VAULT.critique_template
    if vault_path.exists():
        return vault_path.read_text(encoding="utf-8")
    return CRITIQUE_TEMPLATE


def _parse_flags(raw: str) -> list[CritiqueFlag]:
    """Parse the critic's JSON output into :class:`CritiqueFlag` objects.

    Same forgiveness as the deviation parser — strip leading code
    fences, trim prose around the JSON delimiters — so the call
    succeeds even on borderline-malformed responses.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if "```" in text:
            text = text.split("```", 1)[0]
    start = min((text.find(c) for c in "[{" if text.find(c) != -1), default=-1)
    if start > 0:
        text = text[start:]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("flags", [])
    out: list[CritiqueFlag] = []
    for f in payload:
        out.append(
            CritiqueFlag(
                category=f.get("category", "translationese"),
                severity=f.get("severity", "minor"),
                pov_specific=bool(f.get("pov_specific", False)),
                jp_source=f.get("jp_source", ""),
                span=f.get("span", ""),
                suggested_rewrite=f.get("suggested_rewrite", ""),
                notes=f.get("notes", ""),
                violates_rule_id=f.get("violates_rule_id", ""),
            )
        )
    return out


def critique_translation(
    *,
    part_id: str,
    jp: str,
    draft: str,
    model: str | None = None,
    lookup: POVLookup | None = None,
) -> CritiqueResult:
    """Audit ``draft`` for translationese / voice / style drift.

    The critic does **not** see the human reference — it judges the
    draft on its own merits, using the active rules + glossary + style
    profile as the standard for "what good looks like." This is what
    makes inline critique useful at inference time, when no reference
    is available.
    """
    lookup = lookup or load_pov_lookup()
    pov = lookup.pov(part_id)
    chosen_model = model or MODELS.critic
    rules = load_active_rules()
    glossary_entries = load_glossary()
    prompt = Template(_load_critique_template()).safe_substitute(
        part_id=part_id,
        pov=pov,
        jp_source=jp,
        draft=draft,
        active_rules=format_rules_for_prompt(rules),
        glossary=format_glossary_for_prompt(glossary_entries),
        style_profile=format_style_profile_for_prompt(),
    )
    raw = complete(
        model=chosen_model,
        prompt=prompt,
        temperature=0.1,
        # Reasoning tokens count toward this cap. Match the offline
        # deviation extractor (32K) — same workload class, and 16K
        # was empirically not enough at high reasoning.
        max_tokens=32768,
        reasoning_effort=REASONING.critic,  # type: ignore[arg-type]
        json_schema=CRITIQUE_JSON_SCHEMA,
    )
    flags = _parse_flags(raw)
    return CritiqueResult(
        part_id=part_id,
        pov=pov,
        model=chosen_model,
        flags=flags,
        raw=raw,
    )


def revision_required(
    critique: CritiqueResult,
    *,
    severity_threshold: Literal["minor", "major"] = "major",
    minor_threshold: int = 3,
) -> bool:
    """Decide whether a revision pass should run for this critique.

    * ``severity_threshold == "major"`` (default) — revise on any major
      flag, or when the minor count meets ``minor_threshold``.
    * ``severity_threshold == "minor"`` — revise on any flag at all.
    * No flags → no revision.
    """
    if not critique.flags:
        return False
    if severity_threshold == "minor":
        return True
    if critique.major_flags:
        return True
    return len(critique.minor_flags) >= minor_threshold


def format_flags_for_revision(flags: list[CritiqueFlag]) -> str:
    """Render flags as a Markdown table for the revision prompt."""
    if not flags:
        return "_(no flags)_"
    lines = [
        "| # | severity | category | span (in draft) | suggested rewrite | notes |",
        "|---|----------|----------|-----------------|-------------------|-------|",
    ]
    for i, f in enumerate(flags, 1):
        cells = [
            str(i),
            f.severity,
            f.category,
            _md_cell(f.span),
            _md_cell(f.suggested_rewrite),
            _md_cell(f.notes or "—"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_cell(text: str) -> str:
    """Escape pipes + collapse newlines so a string is safe in a table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip() or "—"


__all__ = [
    "CRITIQUE_JSON_SCHEMA",
    "CritiqueFlag",
    "CritiqueResult",
    "critique_translation",
    "format_flags_for_revision",
    "revision_required",
]
