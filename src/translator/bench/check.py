"""Run a checker over a label's translations and persist the verdicts.

Reads the new translation for each ledger chapter, asks the selected
checker whether each known issue is still PRESENT / RESOLVED / UNCLEAR,
and writes the verdicts to ``data/regression/runs/{label}.json``.

The label maps to an output suffix; by default ``bench-{label}`` (what
``bench run`` produces), but any existing suffix can be supplied so you
can check translations you already have on disk (e.g. an old
``v2-rag-deepseek-rules-v11`` run) without re-translating.
"""

from __future__ import annotations

from translator.bench.checkers import get_checker
from translator.bench.ledger import (
    Issue,
    IssueVerdict,
    RunResults,
    issues_for_parts,
    ledger_part_ids,
    save_run,
)
from translator.bench.paths import label_suffix, translation_path
from translator.config import MODELS
from translator.prep.corpus import load_part_jp


class MissingTranslationError(RuntimeError):
    pass


def check_label(
    label: str,
    *,
    suffix: str | None = None,
    checker_name: str = "issue_presence",
    model: str | None = None,
    part_ids: list[str] | None = None,
    include_disabled: bool = False,
    progress=None,
) -> RunResults:
    """Check every (enabled) ledger issue against the label's translations.

    ``suffix`` overrides the output directory suffix (defaults to
    ``bench-{label}``). ``part_ids`` restricts which chapters to check.
    Issues whose chapter has no translation on disk are recorded as
    UNCLEAR with an explanatory reason rather than skipped silently.
    """
    resolved_suffix = suffix if suffix is not None else label_suffix(label)
    targets = set(part_ids) if part_ids is not None else set(ledger_part_ids(include_disabled=include_disabled))
    issues = [
        i
        for i in issues_for_parts(targets, include_disabled=include_disabled)
    ]
    checker = get_checker(checker_name, model=model)

    run = RunResults(
        label=label,
        checker=checker_name,
        model=model or MODELS.judge,
        output_suffix=resolved_suffix,
    )

    # Group by part so each chapter's JP + translation load once.
    by_part: dict[str, list[Issue]] = {}
    for issue in issues:
        by_part.setdefault(issue.part_id, []).append(issue)

    for part_id, part_issues in by_part.items():
        tpath = translation_path(part_id, resolved_suffix)
        if not tpath.exists():
            for issue in part_issues:
                run.results[issue.id] = IssueVerdict(
                    issue_id=issue.id,
                    verdict="UNCLEAR",
                    reason=f"no translation at {tpath}",
                    confidence="low",
                )
            if progress:
                progress(f"{part_id}: MISSING translation ({tpath})")
            continue

        jp_source = load_part_jp(part_id)
        new_translation = tpath.read_text(encoding="utf-8")
        for issue in part_issues:
            if progress:
                progress(f"{part_id} :: {issue.id} [{issue.category}]")
            verdict = checker.check(
                issue, jp_source=jp_source, new_translation=new_translation
            )
            run.results[issue.id] = verdict

    save_run(run)
    return run
