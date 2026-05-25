"""Initialize and inspect the knowledge-vault.

``init_vault()`` is idempotent: running it on an existing vault creates
any missing subdirectories but never overwrites user-edited templates,
glossary, or notes. ``is_initialized()`` tells callers whether they
need to fall back to seed data; ``vault_status()`` gives the CLI a
summary report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from translator.config import PATHS, VAULT
from translator.vault.notes import list_notes
from translator.vault.templates import (
    CLUSTERING_TEMPLATE,
    COMPARISON_TEMPLATE,
    GLOSSARY_SCAFFOLD,
    PROMPT_TEMPLATE,
    STYLE_README,
    VAULT_README,
)


@dataclass
class VaultStatus:
    initialized: bool
    root: Path
    active_rule_count: int
    candidate_rule_count: int
    pruned_rule_count: int
    inactive_rule_count: int
    deviation_round_count: int
    deviation_note_count: int
    evaluation_round_count: int
    glossary_present: bool


def is_initialized(root: Path | None = None) -> bool:
    """Return True iff ``root`` looks like a properly-initialized vault.

    Heuristic: the prompt template exists. We don't require every
    subdirectory because a partially-empty vault (e.g. no rules yet)
    is a normal state.
    """
    root = root or PATHS.knowledge_vault
    return (root / VAULT.prompt_template).exists()


def init_vault(root: Path | None = None, *, overwrite_templates: bool = False) -> Path:
    """Create the vault directory tree and seed it with templates.

    Returns the resolved vault root path. Safe to call multiple times:
    only missing files are written. Pass ``overwrite_templates=True`` to
    forcibly rewrite the prompt and comparison templates from
    :mod:`translator.vault.templates` (used after the canonical templates
    are bumped).
    """
    root = root or PATHS.knowledge_vault
    root.mkdir(parents=True, exist_ok=True)

    for sub in (
        VAULT.deviations,
        VAULT.rules_active,
        VAULT.rules_candidate,
        VAULT.rules_pruned,
        VAULT.rules_inactive,
        VAULT.evaluations,
        VAULT.glossary,
        VAULT.style,
        VAULT.config,
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)

    seeds = [
        (root / VAULT.prompt_template, PROMPT_TEMPLATE),
        (root / VAULT.comparison_template, COMPARISON_TEMPLATE),
        (root / VAULT.clustering_template, CLUSTERING_TEMPLATE),
        (root / VAULT.glossary_file, GLOSSARY_SCAFFOLD),
        (root / VAULT.style_readme, STYLE_README),
        (root / "README.md", VAULT_README),
    ]
    for path, contents in seeds:
        if path.exists() and not overwrite_templates and path.name != "README.md":
            continue
        if path.exists() and path.name == "README.md":
            continue
        path.write_text(contents, encoding="utf-8")

    legacy_profile = root / VAULT.style / "profile.md"
    if legacy_profile.exists():
        legacy_profile.unlink()

    return root


def vault_status(root: Path | None = None) -> VaultStatus:
    """Return a quick summary of the current vault contents."""
    root = root or PATHS.knowledge_vault
    if not root.exists():
        return VaultStatus(
            initialized=False,
            root=root,
            active_rule_count=0,
            candidate_rule_count=0,
            pruned_rule_count=0,
            inactive_rule_count=0,
            deviation_round_count=0,
            deviation_note_count=0,
            evaluation_round_count=0,
            glossary_present=False,
        )

    deviation_root = root / VAULT.deviations
    deviation_rounds = (
        sorted(p for p in deviation_root.iterdir() if p.is_dir())
        if deviation_root.exists()
        else []
    )

    return VaultStatus(
        initialized=is_initialized(root),
        root=root,
        active_rule_count=len(list_notes(root / VAULT.rules_active)),
        candidate_rule_count=len(list_notes(root / VAULT.rules_candidate)),
        pruned_rule_count=len(list_notes(root / VAULT.rules_pruned)),
        inactive_rule_count=len(list_notes(root / VAULT.rules_inactive)),
        deviation_round_count=len(deviation_rounds),
        deviation_note_count=len(list_notes(deviation_root)),
        evaluation_round_count=len(list_notes(root / VAULT.evaluations)),
        glossary_present=(root / VAULT.glossary_file).exists(),
    )
