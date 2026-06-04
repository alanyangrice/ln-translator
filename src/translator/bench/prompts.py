"""Shared loader for bench prompt templates.

All bench prompts live as editable Markdown under ``bench/templates/`` so
they are viewable and tweakable without touching code. Load them through
:func:`load_template` (which takes the bare filename) rather than inlining
prompt strings anywhere.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str) -> str:
    """Return the raw text of ``bench/templates/{name}``."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")
