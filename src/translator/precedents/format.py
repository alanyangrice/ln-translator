"""Render :class:`RetrievalResult` for prompt injection.

The translator/revise/critique templates have a single
``$reference_precedents`` placeholder; this module produces the
Markdown body that replaces it.

When the v2 multi-granular index is in play, the body has two sections:

1. **Phrase precedents** — risk-driven phrase-level matches. Each block
   shows the at-risk JP span (the trigger) and the established
   translator's idiomatic EN rendering plus the literal trap to avoid.
   These are surgical: the translator should adopt the EN rendering
   verbatim or near-verbatim when the same JP pattern appears.
2. **Paragraph precedents** — broader voice / rhythm / register
   matches. Each block shows a paragraph-level JP↔EN pair from the
   parallel corpus that's structurally similar to a paragraph in the
   target chapter.

For v1 results, only the paragraph section renders.
"""

from __future__ import annotations

from collections import defaultdict

from translator.precedents.retrieve import (
    PhrasePrecedent,
    Precedent,
    RetrievalResult,
)

_PLACEHOLDER = (
    "_(no reference precedents available — run "
    "`translator precedents build` to populate this section)_"
)


def _format_paragraph(p: Precedent) -> str:
    """One block per paragraph-level precedent.

    Both the JP and EN sides may span multiple paragraphs (1:2 / 2:1
    shapes), so we render them as indented blocks rather than inline
    arrows.
    """
    jp_lines = [ln.strip() for ln in p.jp_text.splitlines() if ln.strip()] or [
        p.jp_text.strip()
    ]
    en_lines = [ln.strip() for ln in p.en_text.splitlines() if ln.strip()] or [
        p.en_text.strip()
    ]
    jp_block = "\n".join(f"  {line}" for line in jp_lines)
    en_block = "\n".join(f"  {line}" for line in en_lines)
    # v2 paragraphs have length_score=1.0 (LLM quality gate); only
    # bother showing the alignment score for v1 entries where it's
    # informative.
    if p.length_score < 1.0:
        header = (
            f"- [{p.part_id}] shape={p.shape} "
            f"(score {p.query_score:.2f}; align {p.length_score:.2f})"
        )
    else:
        header = (
            f"- [{p.part_id}] shape={p.shape} (score {p.query_score:.2f})"
        )
    return f"{header}\n  JP:\n{jp_block}\n  EN:\n{en_block}"


def _format_phrase_group(query: str, hits: list[PhrasePrecedent]) -> str:
    """Render one risk-triggered group of phrase precedents.

    Multiple precedents may match a single risky JP span (different
    chapters, different category fits, slightly different phrasings).
    We show them as a small bulleted list under the at-risk header.

    The "literal trap" shown is the at-risk scanner's prediction for
    the *new chapter's* JP span (carried on each hit as
    ``query_literal_trap``), not the matched precedent's
    ``literal_alternative`` (which describes a different surface form).
    """
    if not hits:
        return ""
    literal = next((h.query_literal_trap for h in hits if h.query_literal_trap), "")
    category = next((h.query_category for h in hits if h.query_category), hits[0].category)

    lines = [
        f"- AT-RISK JP: `{query}`  (category: {category})",
    ]
    if literal:
        lines.append(f"  - literal trap to avoid: _{literal}_")
    lines.append("  - established renderings:")
    for h in hits:
        note = f" — {h.notes}" if h.notes else ""
        lines.append(
            f"    - [{h.part_id}, score {h.query_score:.2f}] "
            f"`{h.jp_span}` → **{h.en_span}**{note}"
        )
    return "\n".join(lines)


def _group_phrases_by_query(
    phrases: list[PhrasePrecedent],
) -> list[tuple[str, list[PhrasePrecedent]]]:
    """Bucket phrase precedents by their triggering at-risk JP span,
    preserving first-seen order so the prompt reads in chapter sequence.
    """
    groups: dict[str, list[PhrasePrecedent]] = defaultdict(list)
    order: list[str] = []
    for p in phrases:
        key = p.query_jp or p.jp_span
        if key not in groups:
            order.append(key)
        groups[key].append(p)
    return [(k, groups[k]) for k in order]


def format_precedents_for_prompt(result: RetrievalResult | None) -> str:
    """Render a :class:`RetrievalResult` as the body of ``$reference_precedents``.

    Returns the no-precedents placeholder when the result is None or
    empty.
    """
    if result is None or result.is_empty:
        return _PLACEHOLDER

    sections: list[str] = []

    if result.phrases:
        groups = _group_phrases_by_query(result.phrases)
        body = "\n\n".join(_format_phrase_group(q, hits) for q, hits in groups)
        sections.append(
            "## Phrase precedents (risk-driven)\n\n"
            "JP phrases in this chapter that have established idiomatic "
            "renderings in past human translations. When the same JP "
            "pattern appears, adopt the natural English form; do **not** "
            "produce the literal trap shown.\n\n"
            f"{body}"
        )

    if result.paragraphs:
        body = "\n".join(_format_paragraph(p) for p in result.paragraphs)
        sections.append(
            "## Paragraph precedents (voice & rhythm)\n\n"
            "Past JP paragraphs with the established translator's "
            "English rendering. Match these phrasings, paragraph "
            "rhythms, and stylistic choices when the new JP source "
            "has the same shape.\n\n"
            f"{body}"
        )

    if not sections:
        return _PLACEHOLDER

    return "\n\n".join(sections)


__all__ = ["format_precedents_for_prompt"]
