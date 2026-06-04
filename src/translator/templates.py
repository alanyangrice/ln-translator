"""Shared prompt-template loader (DRY core).

Every LLM prompt lives as an editable Markdown file co-located with the
code that uses it, rather than as an inline Python string:

* ``inference/templates/``  — translate.md, revise.md
* ``eval/templates/``       — comparison.md, critique.md, clustering.md, judge.md
* ``style/templates/``      — style_extraction.md
* ``precedents/templates/`` — risk_*.md, precedent_extract_*.md
* ``bench/templates/``      — issue_presence.md, distill_issue.md

Each subpackage exposes a thin ``prompts.py`` that binds these helpers to
its own ``templates/`` directory (so call sites never repeat path logic);
this module is the single implementation they all delegate to.

Some prompts can additionally be overridden by a copy in the knowledge
vault (``knowledge-vault/config/``) so Obsidian edits take effect on the
next run — pass ``vault_rel`` for those. ``base_dir`` is always supplied by
the caller (typically a package ``prompts.py``).
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from translator.config import PATHS


def load_template_with_source(
    name: str,
    base_dir: Path,
    *,
    vault_rel: str | None = None,
) -> tuple[str, str]:
    """Return ``(template_text, source)``.

    If ``vault_rel`` is given and that file exists under the knowledge
    vault, it wins (source ``"vault"``); otherwise the packaged default in
    ``base_dir`` is used (source ``"packaged"``).
    """
    if vault_rel is not None:
        vault_path = PATHS.knowledge_vault / vault_rel
        if vault_path.exists():
            return vault_path.read_text(encoding="utf-8"), "vault"
    return (base_dir / name).read_text(encoding="utf-8"), "packaged"


def load_template(
    name: str,
    base_dir: Path,
    *,
    vault_rel: str | None = None,
) -> str:
    """Return the raw template text (vault override if present, else packaged)."""
    text, _ = load_template_with_source(name, base_dir, vault_rel=vault_rel)
    return text


def render(
    name: str,
    base_dir: Path,
    *,
    vault_rel: str | None = None,
    **substitutions: object,
) -> str:
    """Load a template and apply ``string.Template`` ``$placeholder`` substitution."""
    text = load_template(name, base_dir, vault_rel=vault_rel)
    return Template(text).safe_substitute(**substitutions)


__all__ = ["load_template", "load_template_with_source", "render"]
