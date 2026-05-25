"""Concrete validation checks.

All checks operate on plain strings so they don't depend on whether
the translation came from disk, from a live API call, or from a test
fixture. The aggregator :func:`validate_translation` runs all three and
returns a :class:`ValidationReport` with per-check status and details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from translator.config import THRESHOLDS

CheckStatus = Literal["pass", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregate result for one chapter."""

    part_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def status(self) -> CheckStatus:
        """Overall status: ``fail`` if any check failed, ``warn`` if any warned."""
        if any(c.status == "fail" for c in self.checks):
            return "fail"
        if any(c.status == "warn" for c in self.checks):
            return "warn"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.status == "pass"


# ---------------------------------------------------------------------------
# Dialogue parity
# ---------------------------------------------------------------------------

# Japanese opens dialogue with 「 and closes with 」. Some chapters also
# use 『...』 for nested or quoted dialogue; we count those separately.
_JP_DIALOGUE_OPEN = "「"
_JP_DIALOGUE_CLOSE = "」"


def _count_jp_dialogue_blocks(jp: str) -> int:
    """Count 「...」 dialogue blocks in the Japanese source."""
    open_count = jp.count(_JP_DIALOGUE_OPEN)
    close_count = jp.count(_JP_DIALOGUE_CLOSE)
    return min(open_count, close_count)


def _count_en_dialogue_blocks(en: str) -> int:
    """Count dialogue blocks in the English translation.

    The reference translator preserves 「」 brackets, so we count those.
    If the translation uses straight or curly quotes instead, fall back
    to counting double-quote pairs as a backup signal.
    """
    bracket_count = min(en.count(_JP_DIALOGUE_OPEN), en.count(_JP_DIALOGUE_CLOSE))
    if bracket_count > 0:
        return bracket_count
    # Fallback: count pairs of straight or curly opening/closing double quotes.
    open_q = en.count("\u201c")  # left double quote
    close_q = en.count("\u201d")  # right double quote
    if open_q or close_q:
        return min(open_q, close_q)
    straight = en.count('"')
    return straight // 2


def check_dialogue_parity(jp: str, en: str) -> CheckResult:
    jp_blocks = _count_jp_dialogue_blocks(jp)
    en_blocks = _count_en_dialogue_blocks(en)

    if jp_blocks == 0 and en_blocks == 0:
        return CheckResult(
            name="dialogue_parity",
            status="pass",
            message="No dialogue in either side.",
            details={"jp_blocks": 0, "en_blocks": 0},
        )

    larger = max(jp_blocks, en_blocks, 1)
    skew = abs(jp_blocks - en_blocks) / larger
    threshold = THRESHOLDS.dialogue_parity_max_skew
    status: CheckStatus = "pass" if skew <= threshold else "fail"
    msg = (
        f"JP dialogue blocks: {jp_blocks}; EN dialogue blocks: {en_blocks}; "
        f"skew={skew:.2%} (threshold={threshold:.0%})"
    )
    return CheckResult(
        name="dialogue_parity",
        status=status,
        message=msg,
        details={"jp_blocks": jp_blocks, "en_blocks": en_blocks, "skew": skew},
    )


# ---------------------------------------------------------------------------
# Name frequency
# ---------------------------------------------------------------------------

# Hard-coded for the two leads. Maika/teacher/etc. can be added once the
# corpus exposes more deviations driven by minor characters.
_NAME_PAIRS: list[tuple[str, list[str]]] = [
    # (JP form, EN forms — any of which counts as a hit)
    ("仙台", ["Sendai-san", "Sendai"]),
    ("宮城", ["Miyagi"]),
]


def _count_pattern(text: str, needle: str) -> int:
    return text.count(needle)


def _count_any(text: str, needles: list[str]) -> int:
    return sum(_count_pattern(text, n) for n in needles)


def check_name_frequency(jp: str, en: str) -> CheckResult:
    """Compare JP vs EN occurrence counts for the main characters.

    Threshold bands come from ``THRESHOLDS.name_frequency_{warn,fail}_skew``
    and are calibrated against the human-reference corpus, not set a priori.
    The skew metric is necessarily wide because JP frequently elides
    subjects and EN has to fill them back in; the check primarily exists
    to catch catastrophic name-elision (e.g. an entire character vanishing)
    rather than normal subject-expansion drift.
    """
    rows: list[dict] = []
    worst_skew = 0.0
    for jp_name, en_forms in _NAME_PAIRS:
        jp_n = _count_pattern(jp, jp_name)
        en_n = _count_any(en, en_forms)
        if jp_n == 0 and en_n == 0:
            continue
        denom = max(jp_n, en_n, 1)
        skew = abs(jp_n - en_n) / denom
        worst_skew = max(worst_skew, skew)
        rows.append(
            {"jp_name": jp_name, "en_forms": en_forms, "jp_n": jp_n, "en_n": en_n, "skew": skew}
        )

    if not rows:
        return CheckResult(
            name="name_frequency",
            status="pass",
            message="No tracked character names appeared in either side.",
            details={"rows": []},
        )

    if worst_skew > THRESHOLDS.name_frequency_fail_skew:
        status: CheckStatus = "fail"
    elif worst_skew > THRESHOLDS.name_frequency_warn_skew:
        status = "warn"
    else:
        status = "pass"

    msg_parts = [f"{r['jp_name']}: JP={r['jp_n']} EN={r['en_n']} (skew={r['skew']:.0%})" for r in rows]
    return CheckResult(
        name="name_frequency",
        status=status,
        message="; ".join(msg_parts),
        details={"rows": rows, "worst_skew": worst_skew},
    )


# ---------------------------------------------------------------------------
# Length ratio
# ---------------------------------------------------------------------------

_WS_PAT = re.compile(r"\s+")


def _visible_chars(text: str) -> int:
    """Character count ignoring whitespace runs."""
    return len(_WS_PAT.sub("", text))


def check_length_ratio(jp: str, en: str) -> CheckResult:
    jp_n = _visible_chars(jp)
    en_n = _visible_chars(en)
    if jp_n == 0:
        return CheckResult(
            name="length_ratio",
            status="warn",
            message="JP source is empty after whitespace stripping.",
            details={"jp_chars": 0, "en_chars": en_n},
        )
    ratio = en_n / jp_n
    lo = THRESHOLDS.length_ratio_min
    hi = THRESHOLDS.length_ratio_max
    if lo <= ratio <= hi:
        status: CheckStatus = "pass"
    elif lo * 0.85 <= ratio <= hi * 1.15:
        status = "warn"
    else:
        status = "fail"
    msg = (
        f"EN/JP visible-char ratio: {ratio:.2f} "
        f"(expected {lo:.2f}-{hi:.2f}, jp={jp_n}, en={en_n})"
    )
    return CheckResult(
        name="length_ratio",
        status=status,
        message=msg,
        details={"jp_chars": jp_n, "en_chars": en_n, "ratio": ratio, "min": lo, "max": hi},
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def validate_translation(part_id: str, jp: str, en: str) -> ValidationReport:
    return ValidationReport(
        part_id=part_id,
        checks=[
            check_dialogue_parity(jp, en),
            check_name_frequency(jp, en),
            check_length_ratio(jp, en),
        ],
    )
