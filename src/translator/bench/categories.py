"""Issue category registry.

Categories are kept in one place so adding a new failure mode is a
single edit and every other module (seeding, checking, reporting) can
validate against the same list. The descriptions double as guidance for
the LLM checker when it reasons about whether an issue still applies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    title: str
    description: str


# Derived from the recurring failure modes mined from the user's past
# chapter critiques (parts 230-236). Extend freely — nothing downstream
# hardcodes a fixed set beyond this registry.
CATEGORIES: tuple[Category, ...] = (
    Category(
        "translationese",
        "Translationese / unnatural wording",
        "Phrasing that is grammatically valid but no native novelist would "
        "write; literal calques, stiff rhythm, awkward word choice.",
    ),
    Category(
        "coordination_and",
        "Coordination / 'and'-overuse",
        "Japanese clause order copied into English as run-on 'X, and Y, and "
        "Z' chains instead of being restructured into natural English.",
    ),
    Category(
        "tense",
        "Tense consistency",
        "Present/past tense slips, especially narration drifting out of the "
        "established past-tense voice.",
    ),
    Category(
        "terminology",
        "Terminology consistency",
        "A term rendered inconsistently with how earlier chapters / the "
        "glossary translated it (e.g. order vs request for リクエスト).",
    ),
    Category(
        "untranslated_term",
        "Untranslated / unclear cultural term",
        "A transliterated culinary or cultural term left unexplained where a "
        "gloss or natural English equivalent is needed (nabe, mizuna, etc.).",
    ),
    Category(
        "metaphor",
        "Metaphor / analogy preservation",
        "A vivid metaphor or analogy present in the source (or an earlier "
        "accepted version) that was flattened or dropped.",
    ),
    Category(
        "formatting",
        "Formatting",
        "Quote handling for thoughts/dialogue, stray asterisks/markup, "
        "paragraph spacing and line breaks.",
    ),
    Category(
        "preferred_variant",
        "Preferred earlier rendering",
        "A later version lost something the user explicitly liked in an "
        "earlier version; the liked element should be preserved.",
    ),
    Category(
        "other",
        "Other",
        "Anything not covered by the categories above.",
    ),
)

CATEGORY_KEYS: tuple[str, ...] = tuple(c.key for c in CATEGORIES)

_BY_KEY = {c.key: c for c in CATEGORIES}


def is_valid_category(key: str) -> bool:
    return key in _BY_KEY


def get_category(key: str) -> Category:
    return _BY_KEY.get(key, _BY_KEY["other"])


def describe_categories() -> str:
    """Human/LLM-readable list of categories for prompts and `issues list`."""
    return "\n".join(f"- {c.key}: {c.description}" for c in CATEGORIES)
