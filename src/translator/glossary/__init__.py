"""Glossary loading and prompt formatting.

The glossary is a small, hand-curated table of hard-constraint term
choices ("マフラー" -> "scarf", not "muffler"; "宮城" -> "Miyagi" with
no honorific in Sendai POV; etc.). The source of truth is the Markdown
table at ``knowledge-vault/glossary/glossary.md`` so the user can edit
it in Obsidian. A fresh vault is seeded with a starter table by
``translator vault init`` (see ``GLOSSARY_SCAFFOLD`` in
:mod:`translator.vault.templates`).

The loader returns a list of :class:`GlossaryEntry` and a helper for
formatting them into the prompt.
"""

from __future__ import annotations

from translator.glossary.loader import (
    GlossaryEntry,
    format_for_prompt,
    load_glossary,
)

__all__ = ["GlossaryEntry", "format_for_prompt", "load_glossary"]
