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

# ``translator.inference.provider`` is imported lazily inside
# ``extract_style_profile`` to avoid a circular import: the inference
# package's ``__init__`` pulls in ``inference.prompt``, which in turn
# imports the public surface of ``translator.style`` for prompt
# substitution. Keeping the provider import out of module load order
# breaks the cycle without splitting the package.

# The 16-dimension framework, copied here so the extraction prompt is a
# self-contained unit. Editing a dimension here is a schema change —
# bump ``profile.PROFILE_FORMAT_VERSION`` so existing profiles flag
# themselves for re-extraction.
EXTRACTION_PROMPT = """\
You are a literary style analyst characterizing the prose style of an
English translation of a Japanese web novel. The translation is by a
single human translator working consistently across hundreds of
chapters; your job is to extract that translator's prose signature so
another translator (LLM or human) can match it on future chapters.

# Method

Analyze the provided EN chapters along the 16 dimensions below. For
each dimension write 2–4 sentences of *concrete* observation grounded
in evidence from the corpus. Avoid generic literary descriptors
("evocative", "engaging", "literary") — instead, say specifically what
the writer does and does not do, and quote brief example phrases
(≤10 words) when they illustrate a pattern.

For POV-sensitive dimensions (Tone, Voice, Internal monologue style,
Character voice differentiation), include per-POV subsections
describing how the dimension manifests for Sendai vs. Miyagi narration.
Use these subsection headers verbatim:

  **Sendai:** ...
  **Miyagi:** ...

For non-POV-sensitive dimensions, a single description is enough; only
add per-POV subsections when one POV genuinely diverges from the global
pattern.

If a dimension genuinely doesn't apply (e.g. there is no figurative
language at all), say so explicitly rather than padding.

# 16 Dimensions

## 1. Tone
The emotional attitude behind the words. Describe it as a *combination*
rather than a single adjective — e.g. "dry and resigned with occasional
irritation". Note what emotions are present, what's deliberately
withheld, and how intensity is controlled.

## 2. Voice
The personality of the narrator as expressed through language. What
kind of person sounds like this — what do they notice, what do they
dismiss, how self-aware are they? Voice is tone plus worldview plus
verbal habits.

## 3. Sentence structure
Average sentence length, how complex/compound sentences are used vs.
simple ones, whether fragments appear, and how rhythm varies across a
paragraph. Note patterns like short-short-long alternation or
statement-then-self-correction.

## 4. Word choice
Vocabulary register — casual, formal, technical, poetic. Does the
writer reach for simple concrete words or abstract literary ones? Are
there recurring verbal tics, filler words, or habitual phrases?

## 5. Narrative distance
How close the narrator is to the experience. Are they inside the
moment or reflecting from a distance? Do they editorialize or just
observe? Does the distance shift at key moments?

## 6. Reader trust
How much the prose explains versus implies. Does it spell out emotions
and causality, or leave gaps? Does it restate what's already obvious,
or move on?

## 7. Internal monologue style
How thoughts are rendered — polished narration, fragmented impressions,
rhetorical questions, stream of consciousness, or self-interruption.
Note whether realizations arrive as statements or questions.

## 8. Pacing
How quickly the prose moves through events versus how long it lingers.
Where does it compress time and where does it expand a single moment
into granular detail?

## 9. Figurative language
How often metaphors and similes appear, whether they're conventional or
original, and whether they serve a functional purpose or are
decorative. Note if the style deliberately avoids them.

## 10. Dialogue integration
How dialogue sits within prose — heavy attribution and surrounding
reaction, or bare and unadorned. How much the narrator comments on
what was said versus letting it stand alone.

## 11. Paragraph structure
Average paragraph length, whether single-line paragraphs are used for
emphasis, and how transitions work — abrupt cuts, logical connectors,
or associative jumps.

## 12. Repetition and motif
Whether the prose echoes specific words, phrases, or images
deliberately across passages for thematic reinforcement, or actively
avoids repeating itself.

## 13. Sensory emphasis
Which senses the prose prioritizes and how much physical/bodily detail
appears versus emotional or intellectual processing.

## 14. Tense and temporal framing
Base tense, how time shifts are handled — flashbacks, hypotheticals,
generalizations about the future — and how smoothly the prose moves
between temporal layers.

## 15. Connective tissue
How thoughts link together — explicit logical connectors ("however",
"because"), associative leaps, or bare juxtaposition with no connector
at all.

## 16. Character voice differentiation
Each character should have a distinct way of speaking and thinking
that remains consistent. This includes vocabulary range, sentence
complexity, what they tend to notice or fixate on, how they process
emotions (intellectualizing vs. feeling physically vs. deflecting),
default attitude (confrontational, avoidant, teasing, flat), and
verbal habits in both dialogue and internal thought. When multiple
POVs exist, the prose style itself should shift to reflect whose head
you're in — not just *what* they're thinking about, but *how* they
think. Dialogue should be distinguishable without attribution tags.

For this dimension, give per-character (not just per-POV) subsections:

  **Sendai:** ...
  **Miyagi:** ...

# Output format

Produce one Markdown document with exactly the 16 ``## N. <dimension>``
section headers above, in order. Use the exact dimension *numbers*
above; the extractor relies on them to split the output into one file
per dimension. No preamble, no closing remarks, no JSON. Sub-sections
(the **Sendai:** / **Miyagi:** lines) go directly under the relevant
``##`` heading.

# Materials

The corpus below is in story order. Each chapter is preceded by a
header identifying its part_id and POV. Treat the whole corpus as a
single sample of the translator's prose; the goal is to characterize
the *translator's* style, not any single chapter's content.

$corpus
"""


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
