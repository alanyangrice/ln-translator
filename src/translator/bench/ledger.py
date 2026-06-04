"""Issue ledger + run-result storage.

Design (separation of concerns):

* ``data/regression/issues/{part_id}.jsonl`` — the **ledger**, split one
  file per chapter so issues can be added/edited chapter by chapter. Each
  line is one ``Issue``. Appending a test case is a single line in that
  chapter's file; entries are hand- or CLI-editable.
* ``data/regression/runs/{label}.json`` — **results** of running a
  checker over one bench label. Results never mutate the ledger; status
  history and regressions are computed by joining runs across labels.

The ledger lives under ``data/`` (NOT the knowledge-vault) on purpose:
nothing here is ever loaded into a translation prompt, so it cannot
poison the context with the very phrasings we want to eliminate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from translator.bench.categories import is_valid_category
from translator.config import PATHS

Verdict = Literal["PRESENT", "RESOLVED", "UNCLEAR"]
Severity = Literal["major", "minor"]

VERDICTS: tuple[Verdict, ...] = ("PRESENT", "RESOLVED", "UNCLEAR")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def regression_dir() -> Path:
    return PATHS.data / "regression"


def issues_dir() -> Path:
    return regression_dir() / "issues"


def issue_file(part_id: str) -> Path:
    """Per-chapter ledger file, e.g. ``data/regression/issues/part_230.jsonl``."""
    return issues_dir() / f"{part_id}.jsonl"


def legacy_issues_path() -> Path:
    """Old single-file location, kept only so we can migrate from it."""
    return regression_dir() / "issues.jsonl"


def runs_dir() -> Path:
    return regression_dir() / "runs"


def reports_dir() -> Path:
    return regression_dir() / "reports"


def run_path(label: str) -> Path:
    return runs_dir() / f"{label}.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Issue (ledger entry)
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    """A single regression test case distilled from a past user critique.

    Anchoring is on ``jp_anchor`` (the source span) + ``en_excerpt_original``
    (the offending phrasing) rather than line numbers, which drift between
    translation versions.
    """

    id: str
    part_id: str
    category: str
    user_comment: str
    severity: Severity = "minor"
    first_seen_suffix: str = ""
    # Stable anchors (line numbers are intentionally NOT authoritative).
    jp_anchor: str = ""
    en_excerpt_original: str = ""
    # Acceptance criteria, used by the checker in priority order:
    # preferred_fix -> resolution_guidance -> "is the original gone".
    preferred_fix: str | None = None
    resolution_guidance: str | None = None
    # Provenance so an entry can be audited / re-derived.
    evidence_refs: dict = field(default_factory=dict)
    enabled: bool = True
    source: str = "manual"  # "auto-seed" | "manual"
    created_at: str = field(default_factory=_now)
    # Free-form notes the user can add when editing the ledger by hand.
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> Issue:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty == valid)."""
        problems: list[str] = []
        if not self.id:
            problems.append("missing id")
        if not self.part_id:
            problems.append("missing part_id")
        if not self.user_comment.strip():
            problems.append("empty user_comment")
        if self.severity not in ("major", "minor"):
            problems.append(f"bad severity: {self.severity!r}")
        if not is_valid_category(self.category):
            problems.append(f"unknown category: {self.category!r}")
        return problems


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def _read_issue_file(path: Path) -> list[Issue]:
    if not path.exists():
        return []
    issues: list[Issue] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        issues.append(Issue.from_dict(json.loads(line)))
    return issues


def load_part_issues(part_id: str, *, include_disabled: bool = True) -> list[Issue]:
    issues = _read_issue_file(issue_file(part_id))
    if not include_disabled:
        issues = [i for i in issues if i.enabled]
    return issues


def load_issues(*, include_disabled: bool = True) -> list[Issue]:
    """Concatenate every per-chapter ledger file, ordered by chapter."""
    d = issues_dir()
    issues: list[Issue] = []
    if d.exists():
        for path in sorted(d.glob("*.jsonl")):
            issues.extend(_read_issue_file(path))
    if not include_disabled:
        issues = [i for i in issues if i.enabled]
    return issues


def save_part_issues(part_id: str, issues: list[Issue]) -> Path:
    """Write (or clear) a single chapter's ledger file."""
    path = issue_file(part_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(i.to_json() for i in issues)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return path


def save_issues(issues: list[Issue]) -> Path:
    """Rewrite the whole ledger, one file per chapter.

    Existing ``*.jsonl`` files are cleared first so the on-disk layout
    always matches ``issues`` exactly.
    """
    d = issues_dir()
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob("*.jsonl"):
        stale.unlink()
    by_part: dict[str, list[Issue]] = {}
    for issue in issues:
        by_part.setdefault(issue.part_id, []).append(issue)
    for part_id, part_issues in by_part.items():
        save_part_issues(part_id, part_issues)
    return d


def append_issue(issue: Issue) -> Path:
    """Append one issue to its chapter file, allocating an id if blank."""
    part_issues = load_part_issues(issue.part_id)
    if not issue.id:
        issue.id = next_issue_id(issue.part_id, part_issues)
    if issue.id in {i.id for i in part_issues}:
        raise ValueError(f"issue id already exists: {issue.id}")
    part_issues.append(issue)
    return save_part_issues(issue.part_id, part_issues)


def next_issue_id(part_id: str, issues: list[Issue] | None = None) -> str:
    """Allocate ``issue-{NNN}-{NN}`` keyed on the part number.

    When ``issues`` is omitted, ids are allocated against that chapter's
    file only (ids are namespaced per part, so that's sufficient).
    """
    issues = issues if issues is not None else load_part_issues(part_id)
    num = part_id.split("_")[-1] if "_" in part_id else part_id
    prefix = f"issue-{num}-"
    used = [
        int(i.id[len(prefix):])
        for i in issues
        if i.id.startswith(prefix) and i.id[len(prefix):].isdigit()
    ]
    nxt = (max(used) + 1) if used else 1
    return f"{prefix}{nxt:02d}"


def issues_for_parts(part_ids: set[str], *, include_disabled: bool = False) -> list[Issue]:
    return [
        i
        for i in load_issues(include_disabled=include_disabled)
        if i.part_id in part_ids
    ]


def ledger_part_ids(*, include_disabled: bool = False) -> list[str]:
    seen: list[str] = []
    for i in load_issues(include_disabled=include_disabled):
        if i.part_id not in seen:
            seen.append(i.part_id)
    return seen


# ---------------------------------------------------------------------------
# Run results
# ---------------------------------------------------------------------------

@dataclass
class IssueVerdict:
    issue_id: str
    verdict: Verdict
    evidence: str = ""  # quoted span from the new translation
    reason: str = ""
    confidence: str = "medium"  # high | medium | low

    @classmethod
    def from_dict(cls, data: dict) -> IssueVerdict:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunResults:
    """All checker verdicts for one bench label."""

    label: str
    checker: str
    model: str
    output_suffix: str
    checked_at: str = field(default_factory=_now)
    results: dict[str, IssueVerdict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "checker": self.checker,
            "model": self.model,
            "output_suffix": self.output_suffix,
            "checked_at": self.checked_at,
            "results": {k: asdict(v) for k, v in self.results.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunResults:
        results = {
            k: IssueVerdict.from_dict(v) for k, v in data.get("results", {}).items()
        }
        return cls(
            label=data["label"],
            checker=data.get("checker", "unknown"),
            model=data.get("model", "unknown"),
            output_suffix=data.get("output_suffix", ""),
            checked_at=data.get("checked_at", ""),
            results=results,
        )


def save_run(run: RunResults) -> Path:
    path = run_path(run.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_run(label: str) -> RunResults | None:
    path = run_path(label)
    if not path.exists():
        return None
    return RunResults.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_run_labels() -> list[str]:
    d = runs_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
