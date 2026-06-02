"""Prompt assembler.

Reads the canonical templates from the vault (or falls back to the
in-code constants if the vault hasn't been initialized yet), substitutes
the assembled window + active rules + glossary + reference precedents
+ new chapter, and returns an :class:`AssembledPrompt` carrying both
the rendered text and metadata needed by the eval pipeline.

Precedent retrieval is the only step here that may hit the network: it
embeds the target chapter's JP queries against the on-disk index. The
function gates this call behind ``use_precedents`` and ``index_exists``
so dry-run / offline flows (vault check, tests) don't need a mock. Pre-
fetched precedents can also be passed in via ``precedents=`` so the
caller pays the embedding round-trip once and reuses the result across
the first-pass + revise + critique calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template

from translator.config import PATHS, VAULT
from translator.glossary import format_for_prompt, load_glossary
from translator.inference.window import Window
from translator.precedents import (
    RetrievalResult,
    format_precedents_for_prompt,
    index_exists,
    retrieve_for_part,
)
from translator.prep.corpus import Part, load_part
from translator.style import format_style_profile_for_prompt, load_style_profile
from translator.vault import format_rules_for_prompt, load_active_rules
from translator.vault.templates import PROMPT_TEMPLATE


@dataclass
class AssembledPrompt:
    text: str
    target_part_id: str
    window_part_ids: list[str]
    active_rule_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    template_source: str = "in-code"  # "vault" if loaded from disk
    # Precedents actually injected (for diff/inspection by the eval
    # pipeline). ``None`` when retrieval was disabled or the index
    # wasn't built; an empty result when retrieval ran but matched
    # nothing.
    precedents: RetrievalResult | None = None


def _load_template() -> tuple[str, str]:
    """Return ``(template_string, source_label)``.

    Prefers ``knowledge-vault/config/prompt-template.md`` so user edits
    in Obsidian take effect; falls back to the in-code constant
    otherwise.
    """
    vault_template = PATHS.knowledge_vault / VAULT.prompt_template
    if vault_template.exists():
        return vault_template.read_text(encoding="utf-8"), "vault"
    return PROMPT_TEMPLATE, "in-code"


def _format_window(window: Window) -> str:
    """Render the reference parts as JP-EN pairs, separated by ``---``.

    Human references render with a plain header. AI-translated
    references (extending the window past the human-translated range)
    render with an explicit warning header instructing the model to
    use them for plot continuity only — not as a style anchor — so the
    AI's own stylistic tics don't compound across chapters.
    """
    if not window.parts:
        return "_(no reference parts available — translating cold)_"
    blocks: list[str] = []
    for ref in window.parts:
        ident = ref.entry.id
        pov = ref.entry.pov
        if ref.is_ai_translated:
            label = ref.ai_source_label or "AI-translated"
            header = (
                f"# REFERENCE [{ident}, POV: {pov}] — AI-TRANSLATED ({label})\n\n"
                f"> NOTE: The English text below was produced by an AI translator, "
                f"NOT by the human reference translator. Use it for narrative "
                f"continuity (knowing what just happened in the story) but do "
                f"NOT imitate its stylistic choices, register, idiom selection, "
                f"or sentence rhythm. Only the human reference parts above "
                f"should anchor your translation style. Treat the EN text "
                f"below as plot summary, not as a style example."
            )
        else:
            header = f"# REFERENCE [{ident}, POV: {pov}]"
        block = (
            f"{header}\n\n"
            f"## Japanese\n\n{ref.jp_text.strip()}\n\n"
            f"## English\n\n{ref.en_text.strip()}\n"
        )
        blocks.append(block)
    return "\n---\n\n".join(blocks)


def assemble_prompt(
    target_part_id: str,
    window: Window,
    *,
    target_part: Part | None = None,
    use_precedents: bool = True,
    use_risk_scan: bool = True,
    precedents: RetrievalResult | None = None,
) -> AssembledPrompt:
    """Render the full translation prompt for ``target_part_id``.

    Parameters
    ----------
    use_precedents
        If True (default), retrieve precedents from the on-disk index
        when one exists. If False, skip retrieval entirely and render
        the no-precedents placeholder — used by ``--no-precedents``
        ablations and by dry-run / vault-check paths that must stay
        offline.
    use_risk_scan
        If True (default), and the v2 index is built, run the at-risk
        scanner on the target chapter so phrase-level precedents can
        be retrieved. The scan is cached per-chapter so first call
        pays the LLM round-trip (~3 min on Sonnet) and subsequent
        calls are instant.
    precedents
        Pre-fetched :class:`RetrievalResult` to reuse instead of
        re-querying the index. ``translate_part`` retrieves once and
        passes the same result into the first-pass, revise, and
        critique calls so the embedding round-trip isn't paid three
        times for one chapter.
    """
    template_text, template_source = _load_template()
    template = Template(template_text)

    rules = load_active_rules()
    glossary_entries = load_glossary()
    style_profile = load_style_profile()

    new_part = target_part or load_part(target_part_id)
    notes = list(window.notes)
    if not glossary_entries:
        notes.append("glossary empty — running with no hard-constraint terms")
    if not style_profile.has_content:
        notes.append(
            "style profile empty — run `translator style extract` to populate"
        )
    if template_source == "in-code":
        notes.append("prompt template loaded from in-code default; vault not initialized")

    if precedents is None and use_precedents:
        if index_exists():
            risks = None
            if use_risk_scan:
                from translator.precedents import v2_index_exists

                if v2_index_exists():
                    from translator.precedents.risk import scan_chapter

                    try:
                        risks_result = scan_chapter(target_part_id)
                        risks = risks_result.risks
                        notes.append(
                            f"precedents: at-risk scan flagged "
                            f"{len(risks)} JP span(s) for phrase retrieval"
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        notes.append(
                            f"precedents: at-risk scan failed "
                            f"({type(exc).__name__}: {exc}); falling back "
                            "to paragraph-only retrieval"
                        )
            precedents = retrieve_for_part(target_part_id, risks=risks)
            for n in precedents.notes:
                notes.append(f"precedents: {n}")
        else:
            notes.append(
                "precedents: index not built; "
                "run `translator precedents build`"
            )

    rendered = template.safe_substitute(
        rules=format_rules_for_prompt(rules),
        glossary=format_for_prompt(glossary_entries),
        style_profile=format_style_profile_for_prompt(style_profile),
        reference_precedents=format_precedents_for_prompt(precedents),
        reference_parts=_format_window(window),
        new_part_id=target_part_id,
        new_jp_chapter=new_part.jp_text.strip(),
    )

    return AssembledPrompt(
        text=rendered,
        target_part_id=target_part_id,
        window_part_ids=[r.entry.id for r in window.parts],
        active_rule_ids=[r.id for r in rules],
        notes=notes,
        template_source=template_source,
        precedents=precedents,
    )


__all__ = ["AssembledPrompt", "assemble_prompt"]
