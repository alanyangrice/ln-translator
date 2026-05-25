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
from translator.inference.provider import complete
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.vault.deviations import Deviation, DeviationNote, write_deviation_note
from translator.vault.templates import COMPARISON_TEMPLATE


def _load_comparison_template() -> str:
    vault_path = PATHS.knowledge_vault / VAULT.comparison_template
    if vault_path.exists():
        return vault_path.read_text(encoding="utf-8")
    return COMPARISON_TEMPLATE


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
    prompt = Template(_load_comparison_template()).safe_substitute(
        part_id=part_id,
        pov=pov,
        jp_source=jp,
        llm_translation=llm_translation,
        reference_translation=reference,
    )
    raw = complete(
        model=model,
        prompt=prompt,
        temperature=0.1,
        max_tokens=4096,
        reasoning_effort=REASONING.comparison,  # type: ignore[arg-type]
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
