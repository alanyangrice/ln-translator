"""Glossary loading and prompt formatting.

The glossary is a small, hand-curated table of hard-constraint term
choices ("マフラー" -> "scarf", not "muffler"; "宮城" -> "Miyagi" with
no honorific in Sendai POV; etc.). It's loaded from one of two places,
in priority order:

1. ``knowledge-vault/glossary/glossary.md`` — Markdown table; this is
   the source of truth once the vault is initialized and lets the user
   edit the glossary in Obsidian.
2. ``data/metadata/glossary_seed.json`` — fallback list-of-dicts used
   while the vault is not yet initialized or as initial seed data.

Either way, the loader returns a list of :class:`GlossaryEntry` and a
helper for formatting them into the prompt.
"""

from __future__ import annotations

from translator.glossary.loader import (
    GlossaryEntry,
    format_for_prompt,
    load_glossary,
)

__all__ = ["GlossaryEntry", "format_for_prompt", "load_glossary"]
