"""Loader for style prompt templates (``style/templates/*.md``).

Thin binding over the shared :mod:`translator.templates` loader so prompts
stay co-located with the style code that uses them. Kept dependency-light
(no heavy package imports) so it is safe to import from anywhere.
"""

from __future__ import annotations

from pathlib import Path

from translator.templates import load_template as _load_template
from translator.templates import load_template_with_source as _load_template_with_source
from translator.templates import render as _render

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str, *, vault_rel: str | None = None) -> str:
    return _load_template(name, TEMPLATES_DIR, vault_rel=vault_rel)


def load_template_with_source(name: str, *, vault_rel: str | None = None) -> tuple[str, str]:
    return _load_template_with_source(name, TEMPLATES_DIR, vault_rel=vault_rel)


def render(name: str, *, vault_rel: str | None = None, **substitutions: object) -> str:
    return _render(name, TEMPLATES_DIR, vault_rel=vault_rel, **substitutions)
