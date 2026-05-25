"""Deviation extraction step of the self-evaluation loop.

For each (LLM translation, human reference) pair, ask the comparison
model to produce a structured JSON list of deviations and write the
result to the vault as a per-chapter note. Quality of every downstream
step (rule clustering especially) hinges on this prompt being tight,
so the prompt template lives in the vault and can be edited there.
"""

from __future__ import annotations

import json
from string import Template

from translator.config import MODELS, PATHS, REASONING, VAULT
from translator.glossary import format_for_prompt as format_glossary_for_prompt
from translator.glossary import load_glossary
from translator.inference.provider import complete
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.style import format_style_profile_for_prompt
from translator.vault import format_rules_for_prompt, load_active_rules
from translator.vault.deviations import Deviation, DeviationNote, write_deviation_note
from translator.vault.templates import COMPARISON_TEMPLATE


def _load_comparison_template() -> str:
    vault_path = PATHS.knowledge_vault / VAULT.comparison_template
    if vault_path.exists():
        return vault_path.read_text(encoding="utf-8")
    return COMPARISON_TEMPLATE


DEVIATIONS_JSON_SCHEMA: dict = {
    "name": "deviations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "deviations": {
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
                        "llm_rendering": {"type": "string"},
                        "reference_rendering": {"type": "string"},
                        "notes": {"type": "string"},
                        "violates_rule_id": {
                            "type": "string",
                            "description": (
                                "ID of the active rule this deviation violates "
                                "(e.g. 'rule-000-06'). Empty string if none."
                            ),
                        },
                    },
                    "required": [
                        "category",
                        "severity",
                        "pov_specific",
                        "jp_source",
                        "llm_rendering",
                        "reference_rendering",
                        "notes",
                        "violates_rule_id",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["deviations"],
        "additionalProperties": False,
    },
}


def _parse_deviations(raw: str) -> list[Deviation]:
    """Parse the comparison model's JSON output into :class:`Deviation` objects.

    Tolerates a couple of common formatting issues: a leading ``json``
    code fence, trailing prose, and wrapper objects with a ``deviations``
    array.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip a fenced code block.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if "```" in text:
            text = text.split("```", 1)[0]
    # Strip prose before/after the first JSON delimiter.
    start = min((text.find(c) for c in "[{" if text.find(c) != -1), default=-1)
    if start > 0:
        text = text[start:]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("deviations", [])
    out: list[Deviation] = []
    for d in payload:
        out.append(
            Deviation(
                category=d.get("category", "voice/register"),
                severity=d.get("severity", "minor"),
                pov_specific=bool(d.get("pov_specific", False)),
                jp_source=d.get("jp_source", ""),
                llm_rendering=d.get("llm_rendering", ""),
                reference_rendering=d.get("reference_rendering", ""),
                notes=d.get("notes", ""),
                violates_rule_id=d.get("violates_rule_id", ""),
            )
        )
    return out


def extract_deviations(
    *,
    part_id: str,
    round_number: int,
    jp: str,
    llm_translation: str,
    reference: str,
    model: str | None = None,
    write_to_vault: bool = True,
    lookup: POVLookup | None = None,
) -> DeviationNote:
    """Run the comparison model for one chapter and return the parsed note.

    If ``write_to_vault`` is True (default), the note is also persisted
    to ``deviations/round-NN/part-XXX-deviations.md``.
    """
    lookup = lookup or load_pov_lookup()
    pov = lookup.pov(part_id)
    model = model or MODELS.comparison
    rules = load_active_rules()
    glossary_entries = load_glossary()
    prompt = Template(_load_comparison_template()).safe_substitute(
        part_id=part_id,
        pov=pov,
        jp_source=jp,
        llm_translation=llm_translation,
        reference_translation=reference,
        active_rules=format_rules_for_prompt(rules),
        glossary=format_glossary_for_prompt(glossary_entries),
        style_profile=format_style_profile_for_prompt(),
    )
    raw = complete(
        model=model,
        prompt=prompt,
        temperature=0.1,
        max_tokens=32768,  # JSON array can be long; reasoning tokens count too
        reasoning_effort=REASONING.comparison,  # type: ignore[arg-type]
        json_schema=DEVIATIONS_JSON_SCHEMA,
    )
    deviations = _parse_deviations(raw)
    note = DeviationNote(
        part_id=part_id,
        round_number=round_number,
        pov=pov,
        deviations=deviations,
    )
    if write_to_vault:
        write_deviation_note(note)
    return note


__all__ = ["extract_deviations"]
