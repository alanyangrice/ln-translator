"""Style profile: 16 per-dimension Markdown files plus aggregation.

The profile lives as a directory of one Markdown file per dimension at
``knowledge-vault/style/``:

    01-tone.md
    02-voice.md
    03-sentence-structure.md
    ...
    16-character-voice-differentiation.md

Each file is a self-contained Obsidian-friendly note with YAML
frontmatter recording extraction provenance and a body that starts
with the canonical ``## N. Name`` heading. This split lets the user
hand-edit, link to, or override any single dimension without touching
the rest, and keeps each "type of style" reviewable in isolation.

For prompt injection we concatenate the bodies of all dimension files
(in canonical order) and feed the result to the ``$style_profile``
placeholder. Consumers don't see the per-file split — they get one
unified profile string that looks identical to the LLM output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter

from translator.config import PATHS, VAULT

# Bump this when the 16-dimension framework or extraction prompt is
# incompatibly revised. Profiles produced under an older version still
# load; consumers can decide whether to warn.
PROFILE_FORMAT_VERSION = 1


# Canonical (number, name, slug) for each dimension. The slug forms the
# filename prefix and is *intentionally* fixed here — even if the LLM
# phrases a dimension slightly differently in its output, we always
# write the file at the canonical slug so re-extractions overwrite
# in place rather than producing parallel variants.
DIMENSIONS: tuple[tuple[int, str, str], ...] = (
    (1, "Tone", "tone"),
    (2, "Voice", "voice"),
    (3, "Sentence structure", "sentence-structure"),
    (4, "Word choice", "word-choice"),
    (5, "Narrative distance", "narrative-distance"),
    (6, "Reader trust", "reader-trust"),
    (7, "Internal monologue style", "internal-monologue-style"),
    (8, "Pacing", "pacing"),
    (9, "Figurative language", "figurative-language"),
    (10, "Dialogue integration", "dialogue-integration"),
    (11, "Paragraph structure", "paragraph-structure"),
    (12, "Repetition and motif", "repetition-and-motif"),
    (13, "Sensory emphasis", "sensory-emphasis"),
    (14, "Tense and temporal framing", "tense-and-temporal-framing"),
    (15, "Connective tissue", "connective-tissue"),
    (16, "Character voice differentiation", "character-voice-differentiation"),
)

# Filename pattern: 2-digit number, dash, slug (lowercase + hyphens), .md.
_DIMENSION_FILENAME_RE = re.compile(r"^(?P<num>\d{2})-(?P<slug>[a-z0-9][a-z0-9-]*)\.md$")


def dimension_filename(number: int, slug: str) -> str:
    return f"{number:02d}-{slug}.md"


def canonical_dimension(number: int) -> tuple[int, str, str] | None:
    for n, name, slug in DIMENSIONS:
        if n == number:
            return (n, name, slug)
    return None


@dataclass
class StyleDimension:
    """A single dimension's body + provenance metadata.

    ``body`` is the full Markdown for the dimension *including* the
    ``## N. Name`` heading, so the file is a valid standalone note in
    Obsidian. The aggregator strips the heading-aware structure: it
    just concatenates bodies in dimension order.
    """

    number: int
    name: str
    slug: str
    body: str = ""
    extracted_through: str = ""
    n_chapters: int = 0
    extracted_at: str = ""
    model: str = ""
    version: int = PROFILE_FORMAT_VERSION
    path: Path | None = field(default=None, repr=False)

    @property
    def filename(self) -> str:
        return dimension_filename(self.number, self.slug)

    def to_markdown(self) -> str:
        meta: dict[str, Any] = {
            "version": self.version,
            "dimension_number": self.number,
            "dimension": self.name,
        }
        if self.extracted_through:
            meta["extracted_through"] = self.extracted_through
        if self.n_chapters:
            meta["n_chapters"] = self.n_chapters
        if self.extracted_at:
            meta["extracted_at"] = self.extracted_at
        if self.model:
            meta["model"] = self.model
        post = frontmatter.Post(self.body.strip() + "\n", **meta)
        return frontmatter.dumps(post) + "\n"


@dataclass
class StyleProfile:
    """The full extracted profile, aggregated across all dimensions.

    Construct via :func:`load_style_profile`. The metadata fields
    (``extracted_through`` etc.) are taken from any dimension file's
    frontmatter — they're expected to be identical across all 16
    files written by a single extraction run.
    """

    dimensions: list[StyleDimension] = field(default_factory=list)
    extracted_through: str = ""
    n_chapters: int = 0
    extracted_at: str = ""
    model: str = ""
    version: int = PROFILE_FORMAT_VERSION
    style_dir: Path | None = field(default=None, repr=False)

    @property
    def has_content(self) -> bool:
        """True when at least one dimension has actual extracted content."""
        return any(d.body.strip() for d in self.dimensions) and bool(self.extracted_at)

    @property
    def n_dimensions(self) -> int:
        return len(self.dimensions)

    def render_body(self) -> str:
        """Concatenate dimension bodies (including their ``##`` headings).

        Dimensions are emitted in canonical numerical order so the
        rendered profile reads identically to the original LLM output.
        """
        ordered = sorted(self.dimensions, key=lambda d: d.number)
        return "\n\n".join(d.body.strip() for d in ordered).strip()


def _style_dir() -> Path:
    return PATHS.knowledge_vault / VAULT.style


def _iter_dimension_files(style_dir: Path):
    """Yield ``(path, match)`` pairs for canonical dimension files in dir order."""
    if not style_dir.exists():
        return
    for path in sorted(style_dir.iterdir()):
        if not path.is_file():
            continue
        match = _DIMENSION_FILENAME_RE.match(path.name)
        if match is None:
            continue
        yield path, match


def load_style_profile() -> StyleProfile:
    """Read all dimension files from ``style/`` into one :class:`StyleProfile`.

    Tolerant of the missing-directory and zero-files cases — both
    return an empty placeholder profile with ``has_content`` False so
    consumers can run prompt assembly before the bootstrap.
    """
    style_dir = _style_dir()
    if not style_dir.exists():
        return StyleProfile(style_dir=style_dir)

    dims: list[StyleDimension] = []
    meta_through = ""
    meta_n_chapters = 0
    meta_extracted_at = ""
    meta_model = ""
    meta_version = PROFILE_FORMAT_VERSION

    for path, match in _iter_dimension_files(style_dir):
        try:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = post.metadata
        number = int(meta.get("dimension_number", 0) or int(match.group("num")))
        canonical = canonical_dimension(number)
        name = str(meta.get("dimension", canonical[1] if canonical else match.group("slug")))
        slug = canonical[2] if canonical else match.group("slug")
        body = str(post.content or "").strip()
        if not body:
            continue
        # First valid file with provenance fills the aggregate metadata;
        # later files are expected to match (single-extraction invariant).
        if not meta_extracted_at and meta.get("extracted_at"):
            meta_through = str(meta.get("extracted_through", ""))
            meta_n_chapters = int(meta.get("n_chapters", 0) or 0)
            meta_extracted_at = str(meta.get("extracted_at", ""))
            meta_model = str(meta.get("model", ""))
            meta_version = int(meta.get("version", PROFILE_FORMAT_VERSION) or PROFILE_FORMAT_VERSION)
        dims.append(
            StyleDimension(
                number=number,
                name=name,
                slug=slug,
                body=body,
                extracted_through=str(meta.get("extracted_through", "")),
                n_chapters=int(meta.get("n_chapters", 0) or 0),
                extracted_at=str(meta.get("extracted_at", "")),
                model=str(meta.get("model", "")),
                version=int(meta.get("version", PROFILE_FORMAT_VERSION) or PROFILE_FORMAT_VERSION),
                path=path,
            )
        )

    return StyleProfile(
        dimensions=dims,
        extracted_through=meta_through,
        n_chapters=meta_n_chapters,
        extracted_at=meta_extracted_at,
        model=meta_model,
        version=meta_version,
        style_dir=style_dir,
    )


def write_style_dimensions(
    dimensions: list[StyleDimension],
    *,
    extracted_through: str,
    n_chapters: int,
    model: str,
    extracted_at: str | None = None,
    cleanup_legacy_file: bool = True,
) -> list[Path]:
    """Persist a freshly extracted set of dimension files to ``style/``.

    Each :class:`StyleDimension` is rewritten with the shared
    provenance fields so any single file is self-describing. Returns
    the list of written paths in dimension order.

    When ``cleanup_legacy_file`` is True (default) we remove the legacy
    single-file ``style/profile.md`` placeholder if it's still on disk
    — the directory layout has superseded it.
    """
    style_dir = _style_dir()
    style_dir.mkdir(parents=True, exist_ok=True)
    timestamp = extracted_at or datetime.now(UTC).isoformat(timespec="seconds")

    written: list[Path] = []
    for dim in sorted(dimensions, key=lambda d: d.number):
        dim.extracted_through = extracted_through
        dim.n_chapters = n_chapters
        dim.model = model
        dim.extracted_at = timestamp
        dim.version = PROFILE_FORMAT_VERSION
        path = style_dir / dim.filename
        path.write_text(dim.to_markdown(), encoding="utf-8")
        dim.path = path
        written.append(path)

    if cleanup_legacy_file:
        legacy = style_dir / "profile.md"
        if legacy.exists():
            legacy.unlink()

    return written


_PLACEHOLDER_BODY = (
    "_(no style profile extracted yet — run "
    "`translator style extract` to populate this section)_"
)


def format_style_profile_for_prompt(profile: StyleProfile | None = None) -> str:
    """Render the profile body for prompt injection.

    Returns a placeholder line when no dimensions have been extracted,
    so consumers can pass the result straight through
    ``Template.safe_substitute`` without a None check.
    """
    if profile is None:
        profile = load_style_profile()
    if not profile.has_content:
        return _PLACEHOLDER_BODY
    return profile.render_body()
