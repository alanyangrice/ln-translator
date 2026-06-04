"""Loader for bench prompt templates.

All bench prompts live as editable Markdown under ``bench/templates/`` so
they are viewable and tweakable without touching code. This thin wrapper
delegates to the shared :mod:`translator.templates` loader (DRY) but pins
the base directory to the bench-local templates folder.
"""

from __future__ import annotations

from pathlib import Path

from translator.templates import load_template as _load_template

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str) -> str:
    """Return the raw text of ``bench/templates/{name}``."""
    return _load_template(name, TEMPLATES_DIR)
