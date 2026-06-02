"""Glossary file loading + prompt formatting.

Reads the vault Markdown table (``knowledge-vault/glossary/glossary.md``).
Parsing is forgiving so a translator editing the file in Obsidian doesn't
have to worry about strict column ordering as long as the headers match.
A fresh vault is seeded with the same scaffold table by
``translator vault init``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from translator.config import PATHS, VAULT


@dataclass
class GlossaryEntry:
    japanese: str
    english: str
    notes: str = ""


_TABLE_HEADER_PAT = re.compile(
    r"\|\s*Japanese\s*\|\s*English\s*\|\s*Notes\s*\|", re.IGNORECASE
)
_TABLE_ROW_PAT = re.compile(r"\|(?P<cells>.+)\|\s*$")


def _parse_markdown_table(text: str) -> list[GlossaryEntry]:
    lines = text.splitlines()
    entries: list[GlossaryEntry] = []
    in_table = False
    saw_separator = False
    for line in lines:
        if not in_table:
            if _TABLE_HEADER_PAT.search(line):
                in_table = True
                saw_separator = False
            continue
        # Separator row: |---|---|---|
        stripped = line.strip()
        if not saw_separator:
            if re.match(r"\|[\s\-:|]+\|", stripped):
                saw_separator = True
            continue
        m = _TABLE_ROW_PAT.match(line)
        if not m:
            in_table = False  # End of table
            continue
        cells = [c.strip() for c in m.group("cells").split("|")]
        if len(cells) < 2:
            continue
        jp = cells[0]
        en = cells[1] if len(cells) > 1 else ""
        notes = cells[2] if len(cells) > 2 else ""
        if not jp or jp == "—":
            continue
        entries.append(GlossaryEntry(japanese=jp, english=en, notes="" if notes == "—" else notes))
    return entries


def _load_from_vault(vault_glossary: Path) -> list[GlossaryEntry]:
    return _parse_markdown_table(vault_glossary.read_text(encoding="utf-8"))


def load_glossary() -> list[GlossaryEntry]:
    """Load the glossary from the knowledge vault.

    Returns an empty list when the vault hasn't been initialized yet
    (run ``translator vault init``), so callers don't have to
    special-case bootstrap state.
    """
    vault_glossary = PATHS.knowledge_vault / VAULT.glossary_file
    if vault_glossary.exists():
        return _load_from_vault(vault_glossary)
    return []


def format_for_prompt(entries: list[GlossaryEntry]) -> str:
    """Render the glossary as a Markdown table for prompt injection."""
    if not entries:
        return "_(no glossary entries — using model defaults)_"
    lines = ["| Japanese | English | Notes |", "|----------|---------|-------|"]
    for e in entries:
        notes = e.notes if e.notes else "—"
        lines.append(f"| {e.japanese} | {e.english} | {notes} |")
    return "\n".join(lines)
