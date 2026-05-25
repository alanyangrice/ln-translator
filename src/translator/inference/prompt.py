"""Prompt assembler.

Reads the canonical templates from the vault (or falls back to the
in-code constants if the vault hasn't been initialized yet), substitutes
the assembled window + active rules + glossary + new chapter, and
returns an :class:`AssembledPrompt` carrying both the rendered text and
metadata needed by the eval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template

from translator.config import PATHS, VAULT
from translator.glossary import format_for_prompt, load_glossary
from translator.inference.window import Window
from translator.prep.corpus import Part, load_part
from translator.vault import format_rules_for_prompt, load_active_rules
from translator.vault.templates import PROMPT_TEMPLATE


@dataclass
class AssembledPrompt:
    text: str
    target_part_id: str
    window_part_ids: list[str]
    active_rule_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    template_source: str = "in-code"  # "vault" if loaded from disk


def _load_template() -> tuple[str, str]:
    """Return ``(template_string, source_label)``.

    Prefers ``knowledge-vault/config/prompt-template.md`` so user edits
    in Obsidian take effect; falls back to the in-code constant
    otherwise.
    """
    vault_template = PATHS.knowledge_vault / VAULT.prompt_template
    if vault_template.exists():
        return vault_template.read_text(encoding="utf-8"), "vault"
    return PROMPT_TEMPLATE, "in-code"


def _format_window(window: Window) -> str:
    """Render the reference parts as JP-EN pairs, separated by ``---``."""
    if not window.parts:
        return "_(no reference parts available — translating cold)_"
    blocks: list[str] = []
    for ref in window.parts:
        ident = ref.entry.id
        pov = ref.entry.pov
        header = f"# REFERENCE [{ident}, POV: {pov}]"
        block = (
            f"{header}\n\n"
            f"## Japanese\n\n{ref.jp_text.strip()}\n\n"
            f"## English\n\n{ref.en_text.strip()}\n"
        )
        blocks.append(block)
    return "\n---\n\n".join(blocks)


def assemble_prompt(
    target_part_id: str,
    window: Window,
    *,
    target_part: Part | None = None,
) -> AssembledPrompt:
    """Render the full translation prompt for ``target_part_id``."""
    template_text, template_source = _load_template()
    template = Template(template_text)

    rules = load_active_rules()
    glossary_entries = load_glossary()

    new_part = target_part or load_part(target_part_id)
    notes = list(window.notes)
    if not glossary_entries:
        notes.append("glossary empty — running with no hard-constraint terms")
    if template_source == "in-code":
        notes.append("prompt template loaded from in-code default; vault not initialized")

    rendered = template.safe_substitute(
        rules=format_rules_for_prompt(rules),
        glossary=format_for_prompt(glossary_entries),
        reference_parts=_format_window(window),
        new_part_id=target_part_id,
        new_jp_chapter=new_part.jp_text.strip(),
    )

    return AssembledPrompt(
        text=rendered,
        target_part_id=target_part_id,
        window_part_ids=[r.entry.id for r in window.parts],
        active_rule_ids=[r.id for r in rules],
        notes=notes,
        template_source=template_source,
    )


__all__ = ["AssembledPrompt", "assemble_prompt"]
