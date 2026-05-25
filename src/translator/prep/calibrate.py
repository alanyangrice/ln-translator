"""Corpus-level threshold calibration for the per-chapter validators.

Reads every available JP-EN pair, runs each validator, and produces a
distribution summary plus recommended values for the loose thresholds in
:class:`translator.config.Thresholds`. The recommendation strategy:

* ``length_ratio_min/max``: take the 5th / 95th percentile of the
  observed EN-visible-char / JP-visible-char ratios, then widen by a
  small safety margin so future translator-style drift doesn't
  immediately fail the validator.
* ``dialogue_parity_max_skew``: take the 95th percentile of observed
  per-chapter skews. The reference corpus is by definition correctly
  aligned, so anything inside the 95th percentile must be acceptable.
* ``name_frequency``: surface the worst-case observed skew per tracked
  character. The thresholds inside ``check_name_frequency`` are loose
  on purpose (EN naturally expands subject-elided JP); we just print
  the observed band so the user can decide whether to tighten.

This is a one-time corpus-wide calibration. Re-run it whenever the
corpus changes materially (e.g. when a meaningful number of new parts
are scraped).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from translator.prep.corpus import iter_parts
from translator.validation.checks import (
    check_dialogue_parity,
    check_length_ratio,
    check_name_frequency,
)


@dataclass
class _NumericDistribution:
    """Quantile/mean summary of a numeric series."""

    label: str
    n: int
    mean: float
    stdev: float
    minimum: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    maximum: float

    @classmethod
    def fit(cls, label: str, values: list[float]) -> _NumericDistribution:
        if not values:
            return cls(label=label, n=0, mean=0.0, stdev=0.0,
                       minimum=0.0, p05=0.0, p25=0.0, p50=0.0,
                       p75=0.0, p95=0.0, maximum=0.0)
        ordered = sorted(values)
        n = len(ordered)

        def q(p: float) -> float:
            # Standard linear-interp percentile ("type 7"; matches numpy default).
            if n == 1:
                return ordered[0]
            i = p * (n - 1)
            lo = int(i)
            hi = min(lo + 1, n - 1)
            frac = i - lo
            return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

        return cls(
            label=label,
            n=n,
            mean=statistics.fmean(ordered),
            stdev=statistics.pstdev(ordered) if n > 1 else 0.0,
            minimum=ordered[0],
            p05=q(0.05),
            p25=q(0.25),
            p50=q(0.50),
            p75=q(0.75),
            p95=q(0.95),
            maximum=ordered[-1],
        )


@dataclass
class CalibrationReport:
    """Aggregate output of a corpus-wide calibration run."""

    pair_count: int
    length_ratio: _NumericDistribution = field(
        default_factory=lambda: _NumericDistribution.fit("length_ratio", [])
    )
    dialogue_skew: _NumericDistribution = field(
        default_factory=lambda: _NumericDistribution.fit("dialogue_skew", [])
    )
    name_skew_per_character: dict[str, _NumericDistribution] = field(default_factory=dict)
    name_skew_overall: _NumericDistribution = field(
        default_factory=lambda: _NumericDistribution.fit("name_skew_overall", [])
    )

    def length_ratio_recommendation(self, *, margin: float = 0.10) -> tuple[float, float]:
        """Return ``(min, max)`` candidates: 5th / 95th percentile, padded outward."""
        if self.length_ratio.n == 0:
            return (1.0, 3.0)  # safe default
        lo = round(self.length_ratio.p05 * (1 - margin), 2)
        hi = round(self.length_ratio.p95 * (1 + margin), 2)
        return (lo, hi)

    def dialogue_skew_recommendation(self) -> float:
        """Return a recommended ``dialogue_parity_max_skew``: the 95th percentile."""
        if self.dialogue_skew.n == 0:
            return 0.10
        return round(self.dialogue_skew.p95, 3)


def calibrate() -> CalibrationReport:
    """Iterate every translated pair on disk and produce a calibration report."""
    length_ratios: list[float] = []
    dialogue_skews: list[float] = []
    name_skews: dict[str, list[float]] = {}
    name_skews_all: list[float] = []
    pair_count = 0

    for part in iter_parts(only_translated=True, parts_only=True):
        if part.en_text is None:
            continue
        pair_count += 1
        jp = part.jp_text
        en = part.en_text

        lr = check_length_ratio(jp, en)
        ratio = lr.details.get("ratio")
        if isinstance(ratio, (int, float)):
            length_ratios.append(float(ratio))

        dp = check_dialogue_parity(jp, en)
        skew = dp.details.get("skew")
        if isinstance(skew, (int, float)):
            dialogue_skews.append(float(skew))

        nf = check_name_frequency(jp, en)
        rows = nf.details.get("rows") or []
        for row in rows:
            jp_name = str(row.get("jp_name"))
            row_skew = row.get("skew")
            if isinstance(row_skew, (int, float)):
                name_skews.setdefault(jp_name, []).append(float(row_skew))
                name_skews_all.append(float(row_skew))

    return CalibrationReport(
        pair_count=pair_count,
        length_ratio=_NumericDistribution.fit("length_ratio", length_ratios),
        dialogue_skew=_NumericDistribution.fit("dialogue_skew", dialogue_skews),
        name_skew_per_character={
            name: _NumericDistribution.fit(f"name_skew[{name}]", values)
            for name, values in name_skews.items()
        },
        name_skew_overall=_NumericDistribution.fit("name_skew_overall", name_skews_all),
    )


__all__ = ["CalibrationReport", "calibrate"]
