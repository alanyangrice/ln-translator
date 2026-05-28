"""Legacy v1 precedent pipeline (length-DP alignment + paragraph-only RAG).

Kept for reference and as a fallback when the v2 multi-granular index
(``knowledge-vault/precedent-index-v2``) hasn't been built. The v2
pipeline (LLM-extracted phrase + paragraph precedents at
:mod:`translator.precedents.extract` and :mod:`translator.precedents.risk`)
supersedes everything here for the standard translation flow.

Public API stays stable: ``build_index``, ``validate_index``, and the
length-DP alignment helpers can still be imported via
``translator.precedents`` (the package ``__init__`` re-exports them).
"""

from __future__ import annotations

from translator.precedents.legacy.align import (
    ENParagraph,
    JPParagraph,
    ParagraphAlignment,
    align_paragraphs,
    split_en_paragraphs,
    split_jp_paragraphs,
)
from translator.precedents.legacy.index import (
    IndexEntry,
    IndexStats,
    build_index,
    index_exists,
    load_meta,
)
from translator.precedents.legacy.validate import ValidationStats, validate_index

__all__ = [
    "ENParagraph",
    "IndexEntry",
    "IndexStats",
    "JPParagraph",
    "ParagraphAlignment",
    "ValidationStats",
    "align_paragraphs",
    "build_index",
    "index_exists",
    "load_meta",
    "split_en_paragraphs",
    "split_jp_paragraphs",
    "validate_index",
]
