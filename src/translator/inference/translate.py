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
from translator.inference.ai_references import (
    AIReferenceEntry,
    find_recent_for_target,
)
from translator.inference.prompt import AssembledPrompt, assemble_prompt
from translator.inference.provider import (
    DeepSeekReasoningEffort,
    complete,
    detect_provider,
)
from translator.inference.revise import revise_translation
from translator.inference.window import Window, build_ai_reference, build_window
from translator.precedents import (
    RetrievalResult,
    index_exists,
    retrieve_for_part,
)
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
    revise: bool = False,
    critic_model: str | None = None,
    max_revisions: int | None = None,
    revise_severity: Literal["minor", "major"] | None = None,
    revise_minor_threshold: int | None = None,
    use_precedents: bool = True,
    output_suffix: str | None = None,
    ai_references: list[tuple[str, Path]] | None = None,
    auto_ai_references: bool = True,
    ai_reference_limit: int | None = None,
    risk_model: str | None = None,
    deepseek_thinking: bool = False,
    deepseek_reasoning_effort: DeepSeekReasoningEffort = "high",
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
        If True, run the inline critic after the first-pass
        translation and execute a revision pass when the configured
        severity gate fires. Disabled by default: with precedents at
        k=150 the first-pass draft already adopts most of the
        critic's suggested fixes, and the revision round triples the
        per-chapter cost for marginal polish. Enable selectively when
        you want a second pass on flagged chapters.
    critic_model
        Override ``MODELS.critic``. Useful for ablations.
    max_revisions
        Override ``THRESHOLDS.critique_max_revisions``. ``0`` runs the
        critic but never revises (audit-only mode).
    revise_severity / revise_minor_threshold
        Override the gating knobs in ``THRESHOLDS``.
    use_precedents
        If True (default), retrieve paragraph-level precedents from
        the on-disk RAG index and inject them into the translate /
        revise / critique prompts. Pass False for the
        ``--no-precedents`` ablation or when the index hasn't been
        built yet (the prompt assembler also auto-detects a missing
        index, but this flag short-circuits the embedding round-trip
        even earlier).
    ai_references
        Optional list of ``(part_id, en_text_path)`` tuples appended
        to the sliding window as **AI-translated** references. Used
        to extend context past the last human-translated chapter
        (e.g. translating ``part_231`` with a v2-rag-deepseek run of
        ``part_230`` as the most-recent reference). Each appended
        reference renders with an explicit warning header telling the
        model to use it only for plot continuity, not as a style
        anchor, so AI stylistic tics don't compound across chapters.
        When ``auto_ai_references`` is True the manifest entries are
        also folded in (explicit ones take priority, manifest fills
        the rest).
    auto_ai_references
        If True (default), auto-append the most-recent up to
        ``ai_reference_limit`` entries from
        :attr:`PATHS.ai_references_manifest` whose part number is less
        than the target's. Promote drafts with ``translator ai-ref
        promote``. Pass False to disable manifest auto-loading and rely
        on ``ai_references`` (or no AI references at all).
    ai_reference_limit
        Override :attr:`Thresholds.ai_reference_window_size` (default 10).
        Useful for ablations.
    risk_model
        Override the at-risk scanner model
        (:data:`translator.precedents.risk.DEFAULT_RISK_MODEL`,
        currently ``deepseek-v4-pro``). Cache is scoped by model so
        switching scanners doesn't reuse stale results.
    deepseek_thinking / deepseek_reasoning_effort
        DeepSeek V4 thinking controls for translation calls. By default
        the pipeline keeps DeepSeek in non-thinking mode; enable this for
        slower, deeper ``high`` or ``max`` reasoning passes.
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

    # Merge explicit ai_references (from the caller) with manifest-loaded
    # ones. Explicit overrides take priority: if the same part_id is in
    # both, the explicit path wins. Otherwise the manifest fills any
    # remaining slots up to ``ai_reference_limit``.
    merged_ai_refs: list[tuple[str, Path, AIReferenceEntry | None]] = []
    explicit_part_ids: set[str] = set()
    for ai_part_id, ai_en_path in ai_references or ():
        merged_ai_refs.append((ai_part_id, ai_en_path, None))
        explicit_part_ids.add(ai_part_id)

    if auto_ai_references:
        limit = (
            ai_reference_limit
            if ai_reference_limit is not None
            else THRESHOLDS.ai_reference_window_size
        )
        try:
            manifest_entries = find_recent_for_target(
                part_id, limit=limit + len(explicit_part_ids)
            )
        except FileNotFoundError as exc:
            # Stale manifest entry — fail loud rather than silently
            # producing a translation without continuity context.
            raise FileNotFoundError(
                f"auto AI references could not be loaded: {exc}"
            ) from exc
        for entry in manifest_entries:
            if entry.part_id in explicit_part_ids:
                continue
            if len(merged_ai_refs) >= limit + len(explicit_part_ids):
                break
            merged_ai_refs.append((entry.part_id, entry.en_text, entry))

    # Sort merged refs by part number so the most-recent ends up at the
    # end of the window (immediately before the target chapter).
    merged_ai_refs.sort(key=lambda t: int(t[0].split("_")[1]))

    for ai_part_id, ai_en_path, entry in merged_ai_refs:
        ai_ref = build_ai_reference(
            ai_part_id, ai_en_path, lookup=lookup
        )
        window.parts.append(ai_ref)
        provenance = entry.source_suffix if entry else ai_en_path.parent.name
        window.notes.append(
            f"window: appended AI-translated reference {ai_part_id} "
            f"from {provenance}"
            + (" (from manifest)" if entry else " (explicit)")
        )

    # Retrieve precedents once and reuse across translate + revise +
    # critique so the embedding round-trip is paid one time per
    # chapter regardless of how many critic/revision rounds run.
    # When the v2 index is in play, also run the at-risk scanner so
    # phrase-level precedents can be retrieved for the flagged spans.
    precedents: RetrievalResult | None = None
    fetch_precedents = use_precedents and not dry_run and index_exists()
    risks = None
    if fetch_precedents:
        from translator.precedents import v2_index_exists

        if v2_index_exists():
            try:
                from translator.precedents.risk import (
                    DEFAULT_RISK_MODEL,
                    scan_chapter,
                )

                risks_result = scan_chapter(
                    part_id, model=risk_model or DEFAULT_RISK_MODEL
                )
                risks = risks_result.risks
            except Exception:  # pylint: disable=broad-except
                risks = None
        precedents = retrieve_for_part(part_id, risks=risks)

    prompt = assemble_prompt(
        part_id,
        window,
        use_precedents=use_precedents,
        # Already fetched precedents above (with risks if v2);
        # assemble_prompt should not re-scan or re-retrieve.
        use_risk_scan=False,
        precedents=precedents,
    )

    model = model or MODELS.translation
    notes: list[str] = []
    translation: str | None = None
    draft_v1: str | None = None
    critiques: list[CritiqueResult] = []
    revision_count = 0

    if not dry_run:
        provider = detect_provider(model)
        notes.append(f"calling {provider} with model={model}")
        if provider == "deepseek":
            mode = (
                f"thinking={deepseek_reasoning_effort}"
                if deepseek_thinking
                else "thinking=disabled"
            )
            notes.append(f"deepseek: {mode}")
        if fetch_precedents and precedents is not None:
            phrase_count = len(precedents.phrases)
            para_count = len(precedents.paragraphs)
            risk_count = len(risks) if risks else 0
            tag = f"v{precedents.index_version[1:]}" if precedents.index_version else "v1"
            notes.append(
                f"precedents [{precedents.index_version}]: "
                f"{phrase_count} phrase + {para_count} paragraph pair(s) "
                f"injected"
                + (f" (risk scan flagged {risk_count})" if risk_count else "")
            )
        elif use_precedents:
            notes.append("precedents: index not built; skipping retrieval")
        else:
            notes.append("precedents: disabled (--no-precedents)")
        # Literary chapters routinely exceed the provider default
        # (4K output tokens). The longest EN chapter in the corpus is
        # ~4.5K tokens, so 16K gives ~3.5× headroom and is well below
        # Claude Opus 4.7's 32K output cap. Tune via
        # ``THRESHOLDS.translation_max_tokens`` if a future chapter
        # demands more.
        max_tokens = THRESHOLDS.translation_max_tokens
        if (
            provider == "deepseek"
            and deepseek_thinking
            and deepseek_reasoning_effort == "max"
        ):
            # DeepSeek returns a long reasoning trace before the final
            # content. Think Max needs substantially more output headroom
            # than a non-thinking literary translation.
            max_tokens = max(max_tokens, 65536)
        translation = complete(
            model=model,
            prompt=prompt.text,
            max_tokens=max_tokens,
            deepseek_thinking=deepseek_thinking,
            deepseek_reasoning_effort=deepseek_reasoning_effort,
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
                use_precedents=use_precedents,
                precedents=precedents,
                deepseek_thinking=deepseek_thinking,
                deepseek_reasoning_effort=deepseek_reasoning_effort,
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
            output_suffix=output_suffix,
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
    use_precedents: bool = True,
    precedents: RetrievalResult | None = None,
    deepseek_thinking: bool = False,
    deepseek_reasoning_effort: DeepSeekReasoningEffort = "high",
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
        use_precedents=use_precedents,
        precedents=precedents,
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
            use_precedents=use_precedents,
            precedents=precedents,
            deepseek_thinking=deepseek_thinking,
            deepseek_reasoning_effort=deepseek_reasoning_effort,
        )
        revision_count += 1
        current_draft = revised
        next_critique = critique_translation(
            part_id=target_part_id,
            jp=jp_text,
            draft=current_draft,
            model=critic_model,
            lookup=lookup,
            use_precedents=use_precedents,
            precedents=precedents,
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
    output_suffix: str | None = None,
) -> Path:
    dir_name = part_id if not output_suffix else f"{part_id}-{output_suffix}"
    out_dir = PATHS.output / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.md").write_text(prompt.text, encoding="utf-8")
    precedent_meta: dict | None = None
    if prompt.precedents is not None:
        precedent_meta = {
            "paragraph_count": len(prompt.precedents.paragraphs),
            "notes": list(prompt.precedents.notes),
        }
    meta = {
        "target_part_id": part_id,
        "model": model,
        "dry_run": dry_run,
        "window_part_ids": prompt.window_part_ids,
        "active_rule_ids": prompt.active_rule_ids,
        "notes": prompt.notes,
        "template_source": prompt.template_source,
        "revision_count": revision_count,
        "precedents": precedent_meta,
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
