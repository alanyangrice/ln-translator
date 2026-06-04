"""Pluggable checkers.

A *checker* answers one question for one issue against one new translation:
"is this previously-identified problem still PRESENT, now RESOLVED, or
UNCLEAR?" The harness (``check.py``) depends only on the :class:`Checker`
protocol and a name->factory registry, so new judging strategies (e.g. a
pairwise A/B judge, or a cheap regex heuristic for formatting issues) can
be added later without touching the run/report code.

The default :class:`IssuePresenceChecker` is a reference-free LLM judge —
appropriate because chapters 230+ have no human reference, which matches
the user's real workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from string import Template
from typing import Protocol

from translator.bench.categories import get_category
from translator.bench.ledger import Issue, IssueVerdict
from translator.bench.prompts import load_template
from translator.config import MODELS, REASONING
from translator.inference.provider import complete


class Checker(Protocol):
    """Strategy interface. Implementations must be cheap to construct."""

    name: str

    def check(self, issue: Issue, *, jp_source: str, new_translation: str) -> IssueVerdict:
        ...


# ---------------------------------------------------------------------------
# Default: reference-free LLM issue-presence checker
# ---------------------------------------------------------------------------

_VERDICT_SCHEMA: dict = {
    "name": "issue_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["RESOLVED", "PRESENT", "UNCLEAR"]},
            "evidence": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["verdict", "evidence", "reason", "confidence"],
        "additionalProperties": False,
    },
}


@dataclass
class IssuePresenceChecker:
    """LLM judge that decides PRESENT/RESOLVED/UNCLEAR for a single issue.

    ``stabilize`` re-runs the judge once when the first verdict is
    UNCLEAR or low-confidence and keeps the more decisive answer, trading
    a little cost for less noise on borderline cases.
    """

    name: str = "issue_presence"
    model: str | None = None
    stabilize: bool = True

    def _call(self, issue: Issue, *, jp_source: str, new_translation: str) -> IssueVerdict:
        cat = get_category(issue.category)
        prompt = Template(load_template("issue_presence.md")).safe_substitute(
            category=issue.category,
            category_description=cat.description,
            user_comment=issue.user_comment or "(none given)",
            en_excerpt_original=issue.en_excerpt_original or "(not recorded)",
            jp_anchor=issue.jp_anchor or "(none)",
            preferred_fix=issue.preferred_fix or "(none)",
            resolution_guidance=issue.resolution_guidance or "(none)",
            jp_source=jp_source.strip(),
            new_translation=new_translation.strip(),
        )
        raw = complete(
            model=self.model or MODELS.judge,
            prompt=prompt,
            temperature=0.0,
            # Reasoning tokens count toward max_tokens for high-effort
            # models, so give the same headroom the rubric judge uses.
            max_tokens=16384,
            reasoning_effort=REASONING.judge,  # type: ignore[arg-type]
            json_schema=_VERDICT_SCHEMA,
        )
        payload = json.loads(raw)
        return IssueVerdict(
            issue_id=issue.id,
            verdict=payload.get("verdict", "UNCLEAR"),
            evidence=str(payload.get("evidence", "")),
            reason=str(payload.get("reason", "")),
            confidence=str(payload.get("confidence", "medium")),
        )

    def check(self, issue: Issue, *, jp_source: str, new_translation: str) -> IssueVerdict:
        first = self._call(issue, jp_source=jp_source, new_translation=new_translation)
        if not self.stabilize:
            return first
        if first.verdict != "UNCLEAR" and first.confidence != "low":
            return first
        second = self._call(issue, jp_source=jp_source, new_translation=new_translation)
        # Prefer a decisive, higher-confidence verdict.
        ranked = {"high": 3, "medium": 2, "low": 1}
        if second.verdict == "UNCLEAR":
            return first
        if first.verdict == "UNCLEAR":
            return second
        return second if ranked.get(second.confidence, 0) > ranked.get(first.confidence, 0) else first


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, callable] = {
    "issue_presence": lambda **kw: IssuePresenceChecker(**kw),
}


def available_checkers() -> list[str]:
    return sorted(_REGISTRY)


def get_checker(name: str = "issue_presence", **kwargs) -> Checker:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown checker {name!r}; available: {', '.join(available_checkers())}"
        )
    return _REGISTRY[name](**kwargs)


def register_checker(name: str, factory) -> None:
    """Register a new checker factory (``**kwargs -> Checker``)."""
    _REGISTRY[name] = factory
