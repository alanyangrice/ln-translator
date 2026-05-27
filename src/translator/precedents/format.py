"""Render :class:`RetrievalResult` for prompt injection.

The translator/revise/critique templates have a single
``$reference_precedents`` placeholder; this module produces the
Markdown body that replaces it. Each precedent is one paragraph-level
JP↔EN pair (1:1, 1:2, or 2:1 fan shape) with the provenance tag
``[part_NNN]`` and the cosine query score so the user can debug
retrieval quality from the prompt log.
"""

from __future__ import annotations

from translator.precedents.retrieve import Precedent, RetrievalResult

_PLACEHOLDER = (
    "_(no reference precedents available — run "
    "`translator precedents build` to populate this section)_"
)


def _format_precedent(p: Precedent) -> str:
    """One block per paragraph-level precedent.

    Both the JP and EN sides may span multiple paragraphs (1:2 / 2:1
    shapes), so we render them as indented blocks rather than inline
    arrows. Single-paragraph cases collapse into compact two-line
    blocks; multi-paragraph cases preserve the line breaks.
    """
    jp_lines = [ln.strip() for ln in p.jp_text.splitlines() if ln.strip()] or [
        p.jp_text.strip()
    ]
    en_lines = [ln.strip() for ln in p.en_text.splitlines() if ln.strip()] or [
        p.en_text.strip()
    ]
    jp_block = "\n".join(f"  {line}" for line in jp_lines)
    en_block = "\n".join(f"  {line}" for line in en_lines)
    header = (
        f"- [{p.part_id}] shape={p.shape} "
        f"(score {p.query_score:.2f}; align {p.length_score:.2f})"
    )
    return f"{header}\n  JP:\n{jp_block}\n  EN:\n{en_block}"


def format_precedents_for_prompt(result: RetrievalResult | None) -> str:
    """Render a :class:`RetrievalResult` as the body of ``$reference_precedents``.

    Returns the no-precedents placeholder when the result is None or
    empty.
    """
    if result is None or result.is_empty:
        return _PLACEHOLDER

    body = "\n".join(_format_precedent(p) for p in result.paragraphs)
    return (
        "## Paragraph-level precedents\n\n"
        "Past JP paragraphs with the established translator's English rendering. "
        "Match these phrasings, paragraph rhythms, and stylistic choices when the "
        "new JP source has the same shape.\n\n"
        f"{body}"
    )


__all__ = ["format_precedents_for_prompt"]
