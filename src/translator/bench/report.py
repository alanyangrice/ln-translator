"""Turn run verdicts into a scorecard + regression report.

Joins one label's ``runs/{label}.json`` with an optional baseline label
and the ledger to produce:

* a scorecard (RESOLVED / PRESENT / UNCLEAR counts, overall + by category
  + by chapter),
* REGRESSIONS — issues RESOLVED in the baseline but PRESENT now (the
  over-tuning signal the user cares about),
* IMPROVEMENTS — issues PRESENT in the baseline but RESOLVED now,
* a per-issue table.

Results are immutable per label, so the report is always reconstructable
and a status-history matrix can be built across every run on disk.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from translator.bench.ledger import (
    Issue,
    RunResults,
    list_run_labels,
    load_issues,
    load_run,
    reports_dir,
)


@dataclass
class IssueRow:
    issue: Issue
    verdict: str
    baseline_verdict: str | None
    evidence: str
    reason: str


@dataclass
class Report:
    label: str
    baseline: str | None
    run: RunResults
    rows: list[IssueRow]
    regressions: list[IssueRow] = field(default_factory=list)
    improvements: list[IssueRow] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        return Counter(r.verdict for r in self.rows)


def build_report(label: str, *, baseline: str | None = None) -> Report:
    run = load_run(label)
    if run is None:
        raise FileNotFoundError(
            f"no run results for label {label!r}; run `bench check --label {label}` first."
        )
    base_run = load_run(baseline) if baseline else None
    issues_by_id = {i.id: i for i in load_issues(include_disabled=True)}

    rows: list[IssueRow] = []
    regressions: list[IssueRow] = []
    improvements: list[IssueRow] = []

    for issue_id, verdict in run.results.items():
        issue = issues_by_id.get(issue_id)
        if issue is None:
            continue
        base_v = None
        if base_run is not None and issue_id in base_run.results:
            base_v = base_run.results[issue_id].verdict
        row = IssueRow(
            issue=issue,
            verdict=verdict.verdict,
            baseline_verdict=base_v,
            evidence=verdict.evidence,
            reason=verdict.reason,
        )
        rows.append(row)
        if base_v == "RESOLVED" and verdict.verdict == "PRESENT":
            regressions.append(row)
        elif base_v == "PRESENT" and verdict.verdict == "RESOLVED":
            improvements.append(row)

    rows.sort(key=lambda r: (r.issue.part_id, r.issue.id))
    return Report(
        label=label,
        baseline=baseline,
        run=run,
        rows=rows,
        regressions=regressions,
        improvements=improvements,
    )


def _verdict_tag(v: str) -> str:
    return {"RESOLVED": "[OK]", "PRESENT": "[!!]", "UNCLEAR": "[??]"}.get(v, v)


def render_markdown(report: Report) -> str:
    run = report.run
    counts = report.counts
    total = sum(counts.values())
    lines: list[str] = []
    lines.append(f"# Regression report: {report.label}")
    lines.append("")
    lines.append(f"- checker: `{run.checker}`  model: `{run.model}`")
    lines.append(f"- output suffix: `{run.output_suffix}`")
    lines.append(f"- checked at: {run.checked_at}")
    if report.baseline:
        lines.append(f"- baseline: `{report.baseline}`")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append(f"- total issues checked: {total}")
    lines.append(f"- RESOLVED: {counts.get('RESOLVED', 0)}")
    lines.append(f"- PRESENT: {counts.get('PRESENT', 0)}")
    lines.append(f"- UNCLEAR: {counts.get('UNCLEAR', 0)}")
    lines.append("")

    # By chapter
    by_part: dict[str, Counter] = {}
    by_cat: dict[str, Counter] = {}
    for r in report.rows:
        by_part.setdefault(r.issue.part_id, Counter())[r.verdict] += 1
        by_cat.setdefault(r.issue.category, Counter())[r.verdict] += 1

    lines.append("### By chapter")
    lines.append("")
    lines.append("| chapter | resolved | present | unclear |")
    lines.append("|---|---|---|---|")
    for part_id in sorted(by_part):
        c = by_part[part_id]
        lines.append(
            f"| {part_id} | {c.get('RESOLVED', 0)} | {c.get('PRESENT', 0)} | {c.get('UNCLEAR', 0)} |"
        )
    lines.append("")

    lines.append("### By category")
    lines.append("")
    lines.append("| category | resolved | present | unclear |")
    lines.append("|---|---|---|---|")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        lines.append(
            f"| {cat} | {c.get('RESOLVED', 0)} | {c.get('PRESENT', 0)} | {c.get('UNCLEAR', 0)} |"
        )
    lines.append("")

    if report.baseline:
        lines.append(f"## Regressions vs `{report.baseline}` ({len(report.regressions)})")
        lines.append("")
        if not report.regressions:
            lines.append("None — nothing that was fixed in the baseline came back.")
        else:
            lines.append("Issues that were RESOLVED in the baseline but are PRESENT now:")
            lines.append("")
            for r in report.regressions:
                lines.append(
                    f"- **{r.issue.id}** ({r.issue.part_id}, {r.issue.category}, {r.issue.severity}): "
                    f"{r.issue.user_comment}"
                )
                if r.evidence:
                    lines.append(f"  - now: \"{r.evidence}\" — {r.reason}")
        lines.append("")
        lines.append(f"## Improvements vs `{report.baseline}` ({len(report.improvements)})")
        lines.append("")
        if not report.improvements:
            lines.append("None.")
        else:
            for r in report.improvements:
                lines.append(
                    f"- **{r.issue.id}** ({r.issue.part_id}, {r.issue.category}): "
                    f"{r.issue.user_comment}"
                )
        lines.append("")

    lines.append("## Per-issue detail")
    lines.append("")
    for r in report.rows:
        base = f" (was {r.baseline_verdict})" if r.baseline_verdict else ""
        lines.append(
            f"- {_verdict_tag(r.verdict)} **{r.issue.id}** "
            f"[{r.issue.part_id} / {r.issue.category} / {r.issue.severity}]{base}"
        )
        lines.append(f"  - comment: {r.issue.user_comment}")
        if r.evidence:
            lines.append(f"  - evidence: \"{r.evidence}\"")
        if r.reason:
            lines.append(f"  - reason: {r.reason}")
    lines.append("")
    return "\n".join(lines)


def write_report(report: Report) -> tuple[Path, Path]:
    """Write Markdown + JSON; return both paths."""
    d = reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    name = report.label + (f"-vs-{report.baseline}" if report.baseline else "")
    md_path = d / f"{name}.md"
    json_path = d / f"{name}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    payload = {
        "label": report.label,
        "baseline": report.baseline,
        "checker": report.run.checker,
        "model": report.run.model,
        "output_suffix": report.run.output_suffix,
        "checked_at": report.run.checked_at,
        "counts": dict(report.counts),
        "regressions": [r.issue.id for r in report.regressions],
        "improvements": [r.issue.id for r in report.improvements],
        "rows": [
            {
                "issue_id": r.issue.id,
                "part_id": r.issue.part_id,
                "category": r.issue.category,
                "severity": r.issue.severity,
                "verdict": r.verdict,
                "baseline_verdict": r.baseline_verdict,
                "evidence": r.evidence,
                "reason": r.reason,
            }
            for r in report.rows
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def build_history() -> tuple[list[str], dict[str, dict[str, str]]]:
    """Status-history matrix across every run on disk.

    Returns ``(labels, {issue_id: {label: verdict}})``.
    """
    labels = list_run_labels()
    matrix: dict[str, dict[str, str]] = {}
    for label in labels:
        run = load_run(label)
        if run is None:
            continue
        for issue_id, v in run.results.items():
            matrix.setdefault(issue_id, {})[label] = v.verdict
    return labels, matrix
