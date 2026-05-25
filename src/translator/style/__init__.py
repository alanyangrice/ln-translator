"""Writing-style profile for the translation prompt.

The style profile characterizes the established human translator's
prose along 16 dimensions (tone, voice, sentence structure, narrative
distance, …). Unlike rules — which are *imperative* tactics ("Render X
as Y") — the style profile is a *descriptive* target the model imitates.

The profile is bootstrapped once from the EN reference corpus via
``translator style extract``, persisted to
``knowledge-vault/style/profile.md`` as plain Markdown, and injected
into the translation prompt, the deviation auditor, and the LLM judge
through the ``$style_profile`` placeholder.

Design notes:

* The profile file is human-editable in Obsidian. We don't impose a
  structured schema on it — just a Markdown convention with one
  ``## N. Dimension`` heading per dimension. The whole file gets read
  verbatim into the prompt.
* Loading is intentionally tolerant of an empty or missing file: the
  pipeline degrades gracefully to a "no style profile yet" placeholder
  so a translator can run ``vault check`` before the bootstrap.
"""

from __future__ import annotations

from translator.style.extract import extract_style_profile
from translator.style.profile import (
    DIMENSIONS,
    StyleDimension,
    StyleProfile,
    canonical_dimension,
    format_style_profile_for_prompt,
    load_style_profile,
    write_style_dimensions,
)

__all__ = [
    "DIMENSIONS",
    "StyleDimension",
    "StyleProfile",
    "canonical_dimension",
    "extract_style_profile",
    "format_style_profile_for_prompt",
    "load_style_profile",
    "write_style_dimensions",
]
