"""LLM-driven style profile extraction.

One-shot bootstrap: takes the EN reference corpus (parts 1..N, minus
holdout members), assembles it into a single prompt, and asks the
extraction model to characterize the prose style along the 16
dimensions. The result is parsed into 16 :class:`StyleDimension`
objects (one per file under ``knowledge-vault/style/``) and gets
injected into every subsequent translation/auditor/judge prompt via
the ``$style_profile`` placeholder.

Why a single LLM call rather than per-dimension calls:

* Coherence — sentence structure interacts with pacing, voice
  interacts with internal monologue style, etc. A single call sees
  the whole corpus in one context and can cross-reference patterns.
* Cost — gpt-5.5 with high reasoning is expensive but the call only
  runs once per corpus snapshot (and the bootstrap is the whole
  point of v3's "one-time setup, then sliding window does the work").
* Output integrity — we ask for raw Markdown rather than enforcing a
  JSON schema. The user reviews these files in Obsidian; free-form
  Markdown is friendlier to read and edit than JSON.

Storage shape (one file per dimension, written by
:func:`translator.style.profile.write_style_dimensions`):

    knowledge-vault/style/01-tone.md
    knowledge-vault/style/02-voice.md
    ...
    knowledge-vault/style/16-character-voice-differentiation.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from string import Template

from translator.config import MODELS, REASONING
from translator.prep.corpus import iter_parts
from translator.prep.holdout import load_holdout
from translator.style.profile import (
    DIMENSIONS,
    PROFILE_FORMAT_VERSION,
    StyleDimension,
    canonical_dimension,
)
from translator.style.prompts import load_template

# ``translator.inference.provider`` is imported lazily inside
# ``extract_style_profile`` to avoid a circular import: the inference
# package's ``__init__`` pulls in ``inference.prompt``, which in turn
# imports the public surface of ``translator.style`` for prompt
# substitution. Keeping the provider import out of module load order
# breaks the cycle without splitting the package.

# The 16-dimension framework lives in the prompt template, loaded from
# ``src/translator/templates/style_extraction.md``. Editing a dimension
# there is a schema change — bump ``profile.PROFILE_FORMAT_VERSION`` so
# existing profiles flag themselves for re-extraction.
EXTRACTION_PROMPT = load_template("style_extraction.md")


# Matches level-2 headings of the form `## 7. Internal monologue style`
# at the start of a line. ``re.MULTILINE`` lets us split the whole
# document; the trailing ``re.split`` keeps the headings as separators.
_HEADING_RE = re.compile(
    r"^##\s+(?P<num>\d{1,2})[.:)\s]+(?P<name>.+?)\s*$",
    re.MULTILINE,
)


def parse_dimensions(markdown: str) -> list[StyleDimension]:
    """Split the LLM's Markdown output into one :class:`StyleDimension` per heading.

    Each section runs from one ``## N. Name`` heading to the next (or
    end of document). The returned list is sorted by dimension number,
    de-duplicated (later occurrences win — uncommon but possible if
    the LLM repeats itself), and uses the **canonical** dimension name
    and slug regardless of the LLM's phrasing so filenames stay
    stable across re-extractions.

    Raises ``ValueError`` if the output contains no recognizable
    ``## N. ...`` headings; the caller is expected to surface this so
    a malformed extraction doesn't silently overwrite the vault.
    """
    headings = list(_HEADING_RE.finditer(markdown))
    if not headings:
        raise ValueError(
            "Extraction model returned no ## N. <dimension> headings; "
            "refusing to overwrite the style profile with malformed output."
        )

    by_number: dict[int, StyleDimension] = {}
    for i, match in enumerate(headings):
        number = int(match.group("num"))
        if number < 1 or number > 16:
            continue
        name_emitted = match.group("name").strip()
        canonical = canonical_dimension(number)
        canonical_name = canonical[1] if canonical else name_emitted
        slug = canonical[2] if canonical else _slugify(name_emitted)
        section_start = match.start()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        section = markdown[section_start:section_end].strip()
        # Replace whatever heading the LLM emitted with the canonical
        # form so per-file headings render uniformly in Obsidian.
        section = re.sub(
            r"^##\s+\d{1,2}[.:)\s]+.+?\s*$",
            f"## {number}. {canonical_name}",
            section,
            count=1,
            flags=re.MULTILINE,
        )
        by_number[number] = StyleDimension(
            number=number,
            name=canonical_name,
            slug=slug,
            body=section,
            version=PROFILE_FORMAT_VERSION,
        )

    return [by_number[n] for n in sorted(by_number)]


def _slugify(name: str) -> str:
    """Lowercase + hyphenate a free-form dimension name (fallback only)."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return cleaned or "unnamed"


@dataclass
class ExtractionResult:
    dimensions: list[StyleDimension]
    n_chapters: int
    extracted_through: str
    model: str
    extracted_at: str
    raw_response: str

    @property
    def body(self) -> str:
        """Reconstruct the unified Markdown body (mostly for inspection)."""
        return "\n\n".join(d.body.strip() for d in self.dimensions).strip()


def _gather_corpus(
    *,
    through_part_id: str,
    exclude_holdout: bool,
) -> tuple[list[tuple[str, str, str]], int]:
    """Walk the parallel corpus and collect ``(part_id, pov, en_text)`` tuples.

    Skips:

    * Out-of-scope entries (interludes, side stories, etc.) via
      ``iter_parts(parts_only=True, supported_pov_only=True)``.
    * Entries past ``through_part_id``.
    * Entries without an EN translation.
    * Holdout members when ``exclude_holdout`` is set, so the profile
      doesn't leak content from chapters used for downstream evaluation.
    """
    holdout_ids: set[str] = set()
    if exclude_holdout:
        plan = load_holdout()
        if plan is not None:
            holdout_ids = set(plan.part_ids)

    target_n = int(through_part_id.removeprefix("part_"))
    out: list[tuple[str, str, str]] = []
    skipped_holdout = 0
    for part in iter_parts(only_translated=True):
        if part.part_number is None or part.part_number > target_n:
            continue
        if part.id in holdout_ids:
            skipped_holdout += 1
            continue
        if not part.en_text:
            continue
        out.append((part.id, part.entry.pov, part.en_text.strip()))
    out.sort(key=lambda r: r[0])
    return out, skipped_holdout


def _format_corpus(corpus: list[tuple[str, str, str]]) -> str:
    blocks: list[str] = []
    for part_id, pov, en in corpus:
        blocks.append(f"=== {part_id} (POV: {pov}) ===\n\n{en}")
    return "\n\n".join(blocks)


def extract_style_profile(
    *,
    through_part_id: str = "part_050",
    exclude_holdout: bool = True,
    model: str | None = None,
) -> ExtractionResult:
    """Run the extraction model over the EN corpus and return the parsed result.

    Caller is responsible for persisting via
    :func:`translator.style.profile.write_style_dimensions`. We split
    extraction from persistence so the CLI can preview the parsed
    dimensions before writing.
    """
    from translator.inference.provider import complete

    corpus, _skipped = _gather_corpus(
        through_part_id=through_part_id,
        exclude_holdout=exclude_holdout,
    )
    if not corpus:
        raise RuntimeError(
            f"No EN-translated parts found through {through_part_id}; "
            "run `translator scrape en` first."
        )

    prompt = Template(EXTRACTION_PROMPT).safe_substitute(corpus=_format_corpus(corpus))
    chosen_model = model or MODELS.style_extraction
    raw = complete(
        model=chosen_model,
        prompt=prompt,
        temperature=0.2,
        # Output is long-form Markdown (16 sections × 2–4 sentences,
        # with reasoning headroom). 32K is generous; reasoning tokens
        # share this budget on the OpenAI side.
        max_tokens=32768,
        reasoning_effort=REASONING.style_extraction,  # type: ignore[arg-type]
    )
    dimensions = parse_dimensions(raw)
    expected = {n for n, _, _ in DIMENSIONS}
    seen = {d.number for d in dimensions}
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(
            "Extraction returned an incomplete set of dimensions: "
            f"missing {missing}. Re-run extraction; raw response below.\n\n{raw}"
        )

    return ExtractionResult(
        dimensions=dimensions,
        n_chapters=len(corpus),
        extracted_through=through_part_id,
        model=chosen_model,
        extracted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_response=raw,
    )


__all__ = [
    "EXTRACTION_PROMPT",
    "ExtractionResult",
    "extract_style_profile",
    "parse_dimensions",
]
