"""Evaluation: scoring + self-improvement loop.

Five subroutines, each runnable independently from the CLI:

* :mod:`scores` — COMET + BERTScore over (LLM, reference) pairs.
* :mod:`deviations` — call the comparison model, parse its JSON output,
  and write per-chapter deviation notes to the vault.
* :mod:`clustering` — call the clustering model, write candidate rule
  notes to the vault.
* :mod:`judge` — LLM-as-judge rubric (semantic accuracy, voice fidelity,
  naturalness, style match).
* :mod:`report` — write the round summary note to the vault.

Steps that depend on ML libraries (``unbabel-comet``, ``bert-score``)
require ``uv sync --extra ml`` and are imported lazily so the rest of
the pipeline stays installable on a thin profile.
"""

from __future__ import annotations

from translator.eval.clustering import cluster_into_candidate_rules
from translator.eval.deviations import extract_deviations
from translator.eval.inline_critic import (
    CritiqueFlag,
    CritiqueResult,
    critique_translation,
    format_flags_for_revision,
    revision_required,
)
from translator.eval.judge import judge_translation
from translator.eval.report import RoundSummary, write_round_summary
from translator.eval.scores import bertscore, comet_score

__all__ = [
    "CritiqueFlag",
    "CritiqueResult",
    "RoundSummary",
    "bertscore",
    "cluster_into_candidate_rules",
    "comet_score",
    "critique_translation",
    "extract_deviations",
    "format_flags_for_revision",
    "judge_translation",
    "revision_required",
    "write_round_summary",
]
