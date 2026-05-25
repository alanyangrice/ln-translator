"""Per-chapter pre-window validation.

Three lightweight, deterministic checks run on every translation before
it enters the sliding window for subsequent chapters:

* **Dialogue parity**: count 「」 pairs in JP vs dialogue lines in EN.
* **Name frequency**: verify recurring character names appear with
  similar frequency (large skew suggests speaker misattribution).
* **Length ratio**: EN-character / JP-character ratio should fall in a
  calibrated band; outliers suggest omission or hallucination.

Per the v3 doc on error propagation: a failing translation should be
flagged before being recycled as window context. The aggregator returns
a structured :class:`ValidationReport` so the orchestrator (or an
out-of-band reviewer) can decide how to act on each failure.
"""

from __future__ import annotations

from translator.validation.checks import (
    CheckResult,
    ValidationReport,
    check_dialogue_parity,
    check_length_ratio,
    check_name_frequency,
    validate_translation,
)

__all__ = [
    "CheckResult",
    "ValidationReport",
    "check_dialogue_parity",
    "check_length_ratio",
    "check_name_frequency",
    "validate_translation",
]
