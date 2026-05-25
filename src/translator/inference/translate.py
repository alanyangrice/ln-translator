"""Translation orchestrator.

Glues the window builder, prompt assembler, and provider together. The
``--dry-run`` flag skips the network call and writes only the assembled
prompt to disk; this is the default first sanity check after scaffolding
since it doesn't require API keys or paid tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from translator.config import MODELS, PATHS, THRESHOLDS
from translator.inference.prompt import AssembledPrompt, assemble_prompt
from translator.inference.provider import complete, detect_provider
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
    """Output of a translation call. ``translation`` is None for dry-runs."""

    target_part_id: str
    target_pov: POV
    model: str
    translation: str | None
    prompt: AssembledPrompt
    dry_run: bool
    output_path: Path | None = None
    notes: list[str] = field(default_factory=list)


def translate_part(
    part_id: str,
    *,
    model: str | None = None,
    dry_run: bool = False,
    write_output: bool = True,
    holdout: HoldoutPlan | None | object = ...,  # sentinel for "auto-load"
    lookup: POVLookup | None = None,
) -> TranslationResult:
    """Translate ``part_id`` end-to-end.

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
        If True, write the prompt (and translation if not a dry run) to
        ``data/output/{part_id}/``. Disable for unit tests.
    holdout
        ``None`` to ignore the holdout, a :class:`HoldoutPlan` to skip
        the listed parts as window candidates, or the default sentinel
        which auto-loads ``data/metadata/holdout.json`` if present.
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

    output_path: Path | None = None
    if write_output:
        output_path = _write_output(part_id, prompt, translation, model, dry_run)

    return TranslationResult(
        target_part_id=part_id,
        target_pov=lookup.pov(part_id),
        model=model,
        translation=translation,
        prompt=prompt,
        dry_run=dry_run,
        output_path=output_path,
        notes=notes,
    )


def _write_output(
    part_id: str,
    prompt: AssembledPrompt,
    translation: str | None,
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
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if translation is not None:
        (out_dir / "translation.en.txt").write_text(translation, encoding="utf-8")
    return out_dir


__all__ = ["TranslationResult", "translate_part"]
