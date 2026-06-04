"""Critique-driven regression bench.

Turns the user's own past chapter critiques into a regression suite that
lives OUTSIDE the knowledge-vault (so it never enters a translation
prompt). The flow:

1. ``seed``  — mine past chats into ``data/regression/issues.jsonl``.
2. ``run``   — re-translate the ledger's chapters under a label.
3. ``check`` — ask a pluggable checker whether each known issue is still
   PRESENT / RESOLVED / UNCLEAR; write ``runs/{label}.json``.
4. ``report``— scorecard + regressions (resolved-before-now-present) vs a
   baseline label.

Design goals: adding a test case is one JSONL line; swapping the judging
method is implementing the :class:`~translator.bench.checkers.Checker`
protocol; editing prompts is editing a template file.
"""

from __future__ import annotations

from translator.bench.check import check_label
from translator.bench.checkers import Checker, available_checkers, get_checker
from translator.bench.ledger import (
    Issue,
    IssueVerdict,
    RunResults,
    append_issue,
    load_issues,
    load_part_issues,
    load_run,
    save_issues,
    save_part_issues,
)
from translator.bench.report import build_report, build_history, render_markdown, write_report
from translator.bench.run import run_bench
from translator.bench.seed import build_seed_issues

__all__ = [
    "Checker",
    "Issue",
    "IssueVerdict",
    "RunResults",
    "append_issue",
    "available_checkers",
    "build_history",
    "build_report",
    "build_seed_issues",
    "check_label",
    "get_checker",
    "load_issues",
    "load_part_issues",
    "load_run",
    "render_markdown",
    "run_bench",
    "save_issues",
    "save_part_issues",
    "write_report",
]
