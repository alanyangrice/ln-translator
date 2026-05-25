"""Knowledge-vault read/write helpers.

The v3 architecture stores deviation notes, rule notes, glossary, and
evaluation summaries as plain Markdown with YAML frontmatter under
``knowledge-vault/`` so the same content is browsable in Obsidian *and*
parseable by the pipeline.

Design notes:

* All read paths gracefully handle a non-existent vault — callers don't
  need to special-case "vault not yet initialized". For example,
  :func:`rules.load_active_rules` returns ``[]`` if the vault is missing.
* All write paths assume the vault has been initialized via
  :func:`init.init_vault`. Calling them on an uninitialized vault is a
  ``RuntimeError`` so we never silently scatter half-built state.
* The vault is intended to be its own git repository tracked separately
  from the code, but it can also live as a subdirectory of this repo.
  See ``init.init_vault`` for the layout.
"""

from __future__ import annotations

from translator.vault.deviations import DeviationNote, write_deviation_note
from translator.vault.init import init_vault, is_initialized, vault_status
from translator.vault.notes import Note, read_note, write_note
from translator.vault.rules import (
    Rule,
    RuleState,
    load_active_rules,
    load_pruned_rules,
    load_rules_by_state,
    promote_rule,
    write_rule,
)

__all__ = [
    "DeviationNote",
    "Note",
    "Rule",
    "RuleState",
    "init_vault",
    "is_initialized",
    "load_active_rules",
    "load_pruned_rules",
    "load_rules_by_state",
    "promote_rule",
    "read_note",
    "vault_status",
    "write_deviation_note",
    "write_note",
    "write_rule",
]
