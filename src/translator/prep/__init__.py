"""Corpus preparation: POV lookup, stratified holdout, parallel loaders, calibration.

The v3 pipeline doesn't ship a chunker or train.jsonl builder by default —
those would belong to a future fine-tuning escalation. What it *does* need is:

* A reliable mapping from part id -> POV / chapter / volume so the prompt
  assembler can short-circuit on Maika chapters and the holdout picker
  can stratify by POV.
* A deterministic ~30-part stratified test set whose membership is
  written to ``data/metadata/holdout.json`` and never used as window
  context during evaluation.
* Convenience loaders for the JP and EN sides of any part.
* A one-time calibration pass that fits validator thresholds (length
  ratio, dialogue parity skew) to the observed corpus distribution.
"""

from __future__ import annotations

from translator.prep.calibrate import CalibrationReport, calibrate
from translator.prep.corpus import (
    Part,
    iter_parts,
    load_part,
    load_part_en,
    load_part_jp,
)
from translator.prep.holdout import HoldoutPlan, build_holdout, load_holdout
from translator.prep.pov import (
    POVDetection,
    POVLookup,
    detect_pov_from_disk,
    detect_pov_from_part_content,
    load_pov_lookup,
)

__all__ = [
    "CalibrationReport",
    "HoldoutPlan",
    "POVDetection",
    "POVLookup",
    "Part",
    "build_holdout",
    "calibrate",
    "detect_pov_from_disk",
    "detect_pov_from_part_content",
    "iter_parts",
    "load_holdout",
    "load_part",
    "load_part_en",
    "load_part_jp",
    "load_pov_lookup",
]
