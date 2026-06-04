"""Re-translate the ledger's chapters under a bench label.

A bench *label* names a configuration snapshot (typically the rule
version you're testing, e.g. ``v18``). ``run_bench`` re-translates every
chapter referenced by the ledger into ``data/output/{part}-bench-{label}/``
using the normal pipeline, so the only thing that varies between two
labels is whatever you changed in the vault.

The ledger is never passed to ``translate_part`` — the translator never
sees the issue list, preserving the no-context-poisoning constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

from translator.bench.ledger import ledger_part_ids
from translator.bench.paths import label_suffix, translation_path
from translator.inference import UnsupportedTargetError, translate_part


@dataclass
class BenchRunItem:
    part_id: str
    status: str  # "translated" | "skipped" | "error"
    output_suffix: str
    detail: str = ""


def run_bench(
    label: str,
    *,
    part_ids: list[str] | None = None,
    model: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    use_precedents: bool = True,
    deepseek_thinking: bool = False,
    deepseek_reasoning_effort: str = "high",
    progress=None,
) -> list[BenchRunItem]:
    """Translate each ledger chapter under ``bench-{label}``.

    ``part_ids`` overrides the chapters to run (defaults to every part the
    ledger references). Existing outputs are skipped unless ``force``.
    ``progress`` is an optional ``callable(str)`` for per-chapter logging.
    """
    suffix = label_suffix(label)
    targets = part_ids if part_ids is not None else ledger_part_ids()
    items: list[BenchRunItem] = []

    for part_id in targets:
        out = translation_path(part_id, suffix)
        if out.exists() and not force and not dry_run:
            items.append(BenchRunItem(part_id, "skipped", suffix, "output exists (use --force)"))
            if progress:
                progress(f"skip {part_id} (exists)")
            continue
        try:
            if progress:
                progress(f"translating {part_id} -> {suffix}")
            result = translate_part(
                part_id,
                model=model,
                dry_run=dry_run,
                revise=False,
                use_precedents=use_precedents,
                output_suffix=suffix,
                deepseek_thinking=deepseek_thinking,
                deepseek_reasoning_effort=deepseek_reasoning_effort,  # type: ignore[arg-type]
            )
            detail = result.model + (" (dry-run)" if dry_run else "")
            items.append(BenchRunItem(part_id, "translated", suffix, detail))
        except UnsupportedTargetError as exc:
            items.append(BenchRunItem(part_id, "error", suffix, str(exc)))
            if progress:
                progress(f"error {part_id}: {exc}")
    return items
