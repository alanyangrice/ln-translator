"""Translation orchestrator.

Glues the window builder, prompt assembler, provider, and the inline
critic + revision loop together. The high-level flow per chapter:

1. Build the sliding window and assemble the first-pass prompt.
2. Call the translator model. Persist as ``draft_v1``.
3. (Optional, default on) Call the inline critic against
   ``(jp, draft_v1)``. Persist ``critique.json``.
4. If the critic emits flags above the configured severity gate, run
   one (or more) revision pass(es): assemble a revision prompt that
   includes the previous draft + flags, call the translator again,
   re-critique. Persist intermediate drafts at each iteration.
5. Final translation file is the latest draft.

``--dry-run`` short-circuits steps 2–4 and writes only the assembled
prompt; ``--no-revise`` short-circuits steps 3–4 and writes the
first-pass draft as the final translation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from translator.config import MODELS, PATHS, THRESHOLDS
from translator.eval.inline_critic import (
    CritiqueResult,
    critique_translation,
    revision_required,
)
from translator.inference.prompt import AssembledPrompt, assemble_prompt
from translator.inference.provider import complete, detect_provider
from translator.inference.revise import revise_translation
from translator.inference.window import Window, build_window
from translator.prep.holdout import HoldoutPlan, load_holdout
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.scraper.models import POV, SUPPORTED_POVS


class UnsupportedTargetError(ValueError):
    """Raised when the requested target is outside the v3 pipeline scope.

    The pipeline only translates numbered ``part`` entries with a
    miyagi or sendai POV. Maika side-stories, alternating-POV chapters,
    interludes, extras, and bonus content are intentionally excluded.
    """


@dataclass
class TranslationResult:
    """Output of a translation call. ``translation`` is None for dry-runs.

    For revision-enabled runs:

    * ``translation`` — final draft after all revision passes.
    * ``draft_v1`` — first-pass draft, preserved for diffing.
    * ``critiques`` — one :class:`CritiqueResult` per round (round 0
      is the audit of ``draft_v1``; round 1+ audit each revised draft
      if more than one revision pass runs).
    * ``revision_count`` — how many revision passes actually executed.
    """

    target_part_id: str
    target_pov: POV
    model: str
    translation: str | None
    prompt: AssembledPrompt
    dry_run: bool
    output_path: Path | None = None
    notes: list[str] = field(default_factory=list)
    draft_v1: str | None = None
    critiques: list[CritiqueResult] = field(default_factory=list)
    revision_count: int = 0


def translate_part(
    part_id: str,
    *,
    model: str | None = None,
    dry_run: bool = False,
    write_output: bool = True,
    holdout: HoldoutPlan | None | object = ...,  # sentinel for "auto-load"
    lookup: POVLookup | None = None,
    revise: bool = True,
    critic_model: str | None = None,
    max_revisions: int | None = None,
    revise_severity: Literal["minor", "major"] | None = None,
    revise_minor_threshold: int | None = None,
) -> TranslationResult:
    """Translate ``part_id`` end-to-end with optional critic + revision loop.

    Parameters
    ----------
    part_id
        The id (e.g. ``part_004``) to translate. Must be a numbered
        ``part`` entry with a supported POV; raises
        :class:`UnsupportedTargetError` otherwise.
    model
        Override ``MODELS.translation``. Provider is auto-detected.
    dry_run
        If True, skip the network call and only assemble the prompt.
    write_output
        If True, write artifacts (prompt, drafts, critique, meta) to
        ``data/output/{part_id}/``. Disable for unit tests.
    holdout
        ``None`` to ignore the holdout, a :class:`HoldoutPlan` to skip
        the listed parts as window candidates, or the default sentinel
        which auto-loads ``data/metadata/holdout.json`` if present.
    revise
        If True (default), run the inline critic after the first-pass
        translation and execute a revision pass when the configured
        severity gate fires. Disable for cheap baseline runs or when
        the critic is the thing under test.
    critic_model
        Override ``MODELS.critic``. Useful for ablations.
    max_revisions
        Override ``THRESHOLDS.critique_max_revisions``. ``0`` runs the
        critic but never revises (audit-only mode).
    revise_severity / revise_minor_threshold
        Override the gating knobs in ``THRESHOLDS``.
    """
    lookup = lookup or load_pov_lookup()
    entry = lookup.get(part_id)
    if entry.kind != "part":
        raise UnsupportedTargetError(
            f"{part_id} is a {entry.kind!r} entry; only numbered 'part' entries "
            "are translated by the v3 pipeline."
        )
    if entry.pov not in SUPPORTED_POVS:
        raise UnsupportedTargetError(
            f"{part_id} has POV {entry.pov!r}; only {sorted(SUPPORTED_POVS)} are supported. "
            "Maika POV side-stories and alternating-POV chapters are excluded by design."
        )

    if holdout is ...:
        holdout = load_holdout()

    window: Window = build_window(
        part_id,
        holdout=holdout,  # type: ignore[arg-type]
        lookup=lookup,
    )
    prompt = assemble_prompt(part_id, window)

    model = model or MODELS.translation
    notes: list[str] = []
    translation: str | None = None
    draft_v1: str | None = None
    critiques: list[CritiqueResult] = []
    revision_count = 0

    if not dry_run:
        provider = detect_provider(model)
        notes.append(f"calling {provider} with model={model}")
        # Literary chapters routinely exceed the provider default
        # (4K output tokens). The longest EN chapter in the corpus is
        # ~4.5K tokens, so 16K gives ~3.5× headroom and is well below
        # Claude Opus 4.7's 32K output cap. Tune via
        # ``THRESHOLDS.translation_max_tokens`` if a future chapter
        # demands more.
        translation = complete(
            model=model,
            prompt=prompt.text,
            max_tokens=THRESHOLDS.translation_max_tokens,
        )
        draft_v1 = translation

        if revise:
            translation, critiques, revision_count, revision_notes = _run_critique_revise_loop(
                target_part_id=part_id,
                window=window,
                draft=translation,
                jp=prompt.text,  # we'll use the part's JP source instead
                model=model,
                critic_model=critic_model,
                max_revisions=(
                    max_revisions
                    if max_revisions is not None
                    else THRESHOLDS.critique_max_revisions
                ),
                severity_threshold=revise_severity or THRESHOLDS.critique_revise_severity,  # type: ignore[arg-type]
                minor_threshold=(
                    revise_minor_threshold
                    if revise_minor_threshold is not None
                    else THRESHOLDS.critique_revise_minor_threshold
                ),
                lookup=lookup,
            )
            notes.extend(revision_notes)

    output_path: Path | None = None
    if write_output:
        output_path = _write_output(
            part_id=part_id,
            prompt=prompt,
            translation=translation,
            draft_v1=draft_v1,
            critiques=critiques,
            revision_count=revision_count,
            model=model,
            dry_run=dry_run,
        )

    return TranslationResult(
        target_part_id=part_id,
        target_pov=lookup.pov(part_id),
        model=model,
        translation=translation,
        prompt=prompt,
        dry_run=dry_run,
        output_path=output_path,
        notes=notes,
        draft_v1=draft_v1,
        critiques=critiques,
        revision_count=revision_count,
    )


def _run_critique_revise_loop(
    *,
    target_part_id: str,
    window: Window,
    draft: str,
    jp: str,  # unused; kept for signature symmetry with critique_translation
    model: str,
    critic_model: str | None,
    max_revisions: int,
    severity_threshold: Literal["minor", "major"],
    minor_threshold: int,
    lookup: POVLookup,
) -> tuple[str, list[CritiqueResult], int, list[str]]:
    """Run critic-then-revise iterations until clean or capped.

    Returns ``(final_draft, critiques, revision_count, notes)``.

    Even when ``max_revisions == 0`` we still run one critic pass —
    audit-only mode is useful for baselining without paying the
    revision cost.
    """
    from translator.prep.corpus import load_part_jp

    notes: list[str] = []
    critiques: list[CritiqueResult] = []
    current_draft = draft
    revision_count = 0
    jp_text = load_part_jp(target_part_id)

    # Round 0: audit the first-pass draft.
    critique = critique_translation(
        part_id=target_part_id,
        jp=jp_text,
        draft=current_draft,
        model=critic_model,
        lookup=lookup,
    )
    critiques.append(critique)
    notes.append(
        f"critic[round 0]: {len(critique.major_flags)} major + "
        f"{len(critique.minor_flags)} minor flag(s)"
    )

    # Iterate until we've spent the revision budget or the gate stops firing.
    while revision_count < max_revisions and revision_required(
        critiques[-1],
        severity_threshold=severity_threshold,
        minor_threshold=minor_threshold,
    ):
        revised, _ = revise_translation(
            target_part_id=target_part_id,
            window=window,
            previous_draft=current_draft,
            critique=critiques[-1],
            model=model,
        )
        revision_count += 1
        current_draft = revised
        next_critique = critique_translation(
            part_id=target_part_id,
            jp=jp_text,
            draft=current_draft,
            model=critic_model,
            lookup=lookup,
        )
        critiques.append(next_critique)
        notes.append(
            f"critic[round {revision_count}]: "
            f"{len(next_critique.major_flags)} major + "
            f"{len(next_critique.minor_flags)} minor flag(s)"
        )

    if revision_count == 0 and critiques[0].flags:
        notes.append(
            "critic flagged issues but the gate did not fire; "
            "draft kept unchanged. See critique.json."
        )

    return current_draft, critiques, revision_count, notes


def _write_output(
    *,
    part_id: str,
    prompt: AssembledPrompt,
    translation: str | None,
    draft_v1: str | None,
    critiques: list[CritiqueResult],
    revision_count: int,
    model: str,
    dry_run: bool,
) -> Path:
    out_dir = PATHS.output / part_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.md").write_text(prompt.text, encoding="utf-8")
    meta = {
        "target_part_id": part_id,
        "model": model,
        "dry_run": dry_run,
        "window_part_ids": prompt.window_part_ids,
        "active_rule_ids": prompt.active_rule_ids,
        "notes": prompt.notes,
        "template_source": prompt.template_source,
        "revision_count": revision_count,
        "critic_rounds": [
            {
                "round": i,
                "model": c.model,
                "flag_count": len(c.flags),
                "major_count": len(c.major_flags),
                "minor_count": len(c.minor_flags),
            }
            for i, c in enumerate(critiques)
        ],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if translation is not None:
        (out_dir / "translation.en.txt").write_text(translation, encoding="utf-8")
    if draft_v1 is not None and draft_v1 != translation:
        (out_dir / "draft_v1.en.txt").write_text(draft_v1, encoding="utf-8")
    elif (out_dir / "draft_v1.en.txt").exists() and revision_count == 0:
        # Stale artifact from a prior revised run on the same part.
        (out_dir / "draft_v1.en.txt").unlink()
    if critiques:
        # Persist every round so the user can diff what the revise
        # loop actually fixed. Naming mirrors ``draft_v1.en.txt`` →
        # ``translation.en.txt``: ``critique_v{i+1}.json`` holds the
        # audit of ``draft_v{i+1}``, and ``critique.json`` always
        # holds the audit of the **final** translation (the last
        # round). For a 1-revision run this means:
        #   critique_v1.json  → audit of draft_v1 (round 0)
        #   critique.json     → audit of the final translation
        # For a no-revision run, both files contain the same audit.
        for i, c in enumerate(critiques):
            c.write(out_dir / f"critique_v{i + 1}.json")
        critiques[-1].write(out_dir / "critique.json")
    return out_dir


__all__ = ["TranslationResult", "translate_part"]
