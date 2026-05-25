"""Revision pass for the inline critic + revise loop.

Sister of :mod:`translator.inference.prompt`. The first-pass translator
produces a draft; the inline critic flags translationese / voice
mismatches / style-profile drift; this module assembles a revision
prompt that gives the translator its previous draft + the critic's
flags, and asks for a corrected version.

Why a separate module rather than appending revision sections to
``assemble_prompt``:

* Different lifecycle. The base prompt is rendered once at translation
  time; the revision prompt only exists when a draft has been
  critiqued and a revision is warranted.
* Different placeholders. ``$previous_draft`` and ``$critic_flags``
  don't belong in the first-pass prompt; treating them as optional in
  the base template would make the base template harder to reason
  about.
* Different vault file. ``revise-prompt-template.md`` is hand-editable
  in Obsidian alongside ``prompt-template.md``; the two templates can
  evolve together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template

from translator.config import MODELS, PATHS, THRESHOLDS, VAULT
from translator.eval.inline_critic import (
    CritiqueResult,
    format_flags_for_revision,
)
from translator.glossary import format_for_prompt as format_glossary_for_prompt
from translator.glossary import load_glossary
from translator.inference.prompt import _format_window
from translator.inference.provider import complete, detect_provider
from translator.inference.window import Window
from translator.prep.corpus import Part, load_part
from translator.style import format_style_profile_for_prompt, load_style_profile
from translator.vault import format_rules_for_prompt, load_active_rules
from translator.vault.templates import REVISE_TEMPLATE


@dataclass
class RevisedPrompt:
    """The fully-rendered revision prompt + metadata for logging."""

    text: str
    target_part_id: str
    window_part_ids: list[str]
    active_rule_ids: list[str] = field(default_factory=list)
    n_flags: int = 0
    template_source: str = "in-code"


def _load_revise_template() -> tuple[str, str]:
    vault_template = PATHS.knowledge_vault / VAULT.revise_template
    if vault_template.exists():
        return vault_template.read_text(encoding="utf-8"), "vault"
    return REVISE_TEMPLATE, "in-code"


def assemble_revise_prompt(
    target_part_id: str,
    window: Window,
    *,
    previous_draft: str,
    critique: CritiqueResult,
    target_part: Part | None = None,
) -> RevisedPrompt:
    """Render the revision prompt for ``target_part_id``.

    Same rules + glossary + style + window the first-pass prompt used,
    plus the previous draft and the critic's flags. The translator
    sees its own work and the specific spans to fix; we trust it to
    produce a coherent rewrite that maintains the rest of the chapter.
    """
    template_text, template_source = _load_revise_template()
    template = Template(template_text)

    rules = load_active_rules()
    glossary_entries = load_glossary()
    style_profile = load_style_profile()

    new_part = target_part or load_part(target_part_id)
    rendered = template.safe_substitute(
        rules=format_rules_for_prompt(rules),
        glossary=format_glossary_for_prompt(glossary_entries),
        style_profile=format_style_profile_for_prompt(style_profile),
        reference_parts=_format_window(window),
        new_part_id=target_part_id,
        new_jp_chapter=new_part.jp_text.strip(),
        critic_flags=format_flags_for_revision(critique.flags),
        previous_draft=previous_draft.strip(),
    )

    return RevisedPrompt(
        text=rendered,
        target_part_id=target_part_id,
        window_part_ids=[r.entry.id for r in window.parts],
        active_rule_ids=[r.id for r in rules],
        n_flags=len(critique.flags),
        template_source=template_source,
    )


def revise_translation(
    *,
    target_part_id: str,
    window: Window,
    previous_draft: str,
    critique: CritiqueResult,
    model: str | None = None,
    target_part: Part | None = None,
) -> tuple[str, RevisedPrompt]:
    """Run the revision pass and return ``(revised_text, prompt_metadata)``.

    Caller is responsible for persistence — keeping that here would
    couple the revision module to the on-disk artifact layout, which
    ``translate_part`` is already managing.
    """
    prompt = assemble_revise_prompt(
        target_part_id,
        window,
        previous_draft=previous_draft,
        critique=critique,
        target_part=target_part,
    )
    chosen_model = model or MODELS.translation
    _provider = detect_provider(chosen_model)  # asserted by caller; logged via notes
    revised = complete(
        model=chosen_model,
        prompt=prompt.text,
        max_tokens=THRESHOLDS.translation_max_tokens,
    )
    return revised, prompt


__all__ = [
    "RevisedPrompt",
    "assemble_revise_prompt",
    "revise_translation",
]
