"""Inference: window assembly, prompt construction, translation orchestration.

The flow per part N:

1. :func:`window.build_window` selects reference parts (pure consecutive
   over the supported-POV slice of the corpus).
2. :func:`prompt.assemble_prompt` formats the window + active rules +
   glossary + new JP chapter into the canonical template.
3. :func:`translate.translate_part` calls the configured provider and
   returns the English translation along with prompt metadata for the
   eval pipeline. Raises :class:`UnsupportedTargetError` for non-part
   entries or unsupported POVs.

``--dry-run`` short-circuits step 3 and writes the assembled prompt to
``data/output/`` so it can be inspected without spending API tokens.
"""

from __future__ import annotations

from translator.inference.prompt import AssembledPrompt, assemble_prompt
from translator.inference.revise import RevisedPrompt, assemble_revise_prompt, revise_translation
from translator.inference.translate import (
    TranslationResult,
    UnsupportedTargetError,
    translate_part,
)
from translator.inference.window import ReferencePart, Window, build_window

__all__ = [
    "AssembledPrompt",
    "ReferencePart",
    "RevisedPrompt",
    "TranslationResult",
    "UnsupportedTargetError",
    "Window",
    "assemble_prompt",
    "assemble_revise_prompt",
    "build_window",
    "revise_translation",
    "translate_part",
]
