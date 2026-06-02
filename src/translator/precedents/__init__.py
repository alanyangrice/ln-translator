"""Precedent RAG: cross-lingual retrieval of past JP↔EN translation pairs.

For any new chapter, retrieve JP→EN segments from the parallel corpus
where the JP shape is structurally similar to sentences (and 3-sentence
windows) in the new chapter, and inject them into the
translation/revise/critique prompts as canonical phrasings the model
should imitate.

Mirrors :mod:`translator.style` and :mod:`translator.glossary`: a
one-time mining step over the corpus produces a vault/data artifact,
plus a query-time loader that substitutes a placeholder into existing
templates. Public API:

* :func:`build_index` — populate ``knowledge-vault/precedent-index/`` from
  the parallel corpus (one-time, cheap; v1 length-DP path only).
* :func:`retrieve_for_part` — embed a target chapter's JP queries and
  return the top-K matched precedents at each granularity.
* :func:`format_precedents_for_prompt` — render a :class:`RetrievalResult`
  as the Markdown body of the ``$reference_precedents`` placeholder.

Storage layout is documented in :mod:`translator.precedents.index`.
"""

from __future__ import annotations

from translator.precedents.format import format_precedents_for_prompt
from translator.precedents.legacy.index import (
    IndexEntry,
    IndexStats,
    build_index,
    load_meta,
)
from translator.precedents.legacy.validate import ValidationStats, validate_index
from translator.precedents.retrieve import (
    PhrasePrecedent,
    Precedent,
    RetrievalResult,
    index_exists,
    load_index,
    retrieve_for_part,
    v2_index_exists,
)

__all__ = [
    "IndexEntry",
    "IndexStats",
    "PhrasePrecedent",
    "Precedent",
    "RetrievalResult",
    "ValidationStats",
    "build_index",
    "format_precedents_for_prompt",
    "index_exists",
    "load_index",
    "load_meta",
    "retrieve_for_part",
    "v2_index_exists",
    "validate_index",
]
