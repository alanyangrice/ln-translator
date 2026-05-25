"""Stratified holdout selector for the evaluation test set.

v3 calls for ~30 held-out parts stratified across story position and
POV. The exact members are determined by ``random_seed`` so the holdout
is reproducible across runs and machines, and the result is persisted
to ``data/metadata/holdout.json`` so the prompt assembler knows which
parts to skip when assembling sliding-window context during eval.

Stratification strategy:

1. Restrict to ``kind == "part"`` entries. Interludes, extras, and side
   stories are evaluated separately because they don't follow the same
   numbering or window logic.
2. Split the part axis into 3 buckets (early / mid / late) by part
   number percentile.
3. Inside each bucket, allocate the per-bucket quota proportionally
   between Miyagi and Sendai POVs, with at least one slot reserved for
   each if the bucket has any of that POV.
4. Sample within each (bucket, POV) cell with a seeded RNG.

We do *not* include Maika POV in the auto-stratified split — there are
only ~5 Maika parts and the v3 doc explicitly handles them as a manual
case (the sliding window won't have Maika examples; reference parts
are injected by hand).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from translator.config import PATHS, THRESHOLDS
from translator.prep.pov import POVLookup, load_pov_lookup
from translator.scraper.models import POV, TocEntry


@dataclass
class HoldoutPlan:
    """The selected test set, with metadata about how it was built."""

    part_ids: list[str]
    seed: int
    target_count: int
    per_bucket_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def __contains__(self, part_id: str) -> bool:
        return part_id in self.part_ids

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "target_count": self.target_count,
            "part_ids": list(self.part_ids),
            "per_bucket_counts": self.per_bucket_counts,
        }


def _bucket_for(part_number: int, total_parts: int) -> str:
    """Assign a part to one of three buckets by position in the full corpus."""
    pct = part_number / max(total_parts, 1)
    if pct < 1 / 3:
        return "early"
    if pct < 2 / 3:
        return "mid"
    return "late"


def _allocate(target_count: int) -> dict[str, dict[POV, int]]:
    """Split ``target_count`` evenly across (bucket, POV) cells.

    POV split is approximately 50/50 since the corpus alternates by chapter.
    Buckets get equal shares with any remainder going to the late bucket
    so the most-recent slice gets the strongest signal.
    """
    per_bucket = target_count // 3
    remainder = target_count - per_bucket * 3
    bucket_totals = {"early": per_bucket, "mid": per_bucket, "late": per_bucket + remainder}
    out: dict[str, dict[POV, int]] = {}
    for bucket, total in bucket_totals.items():
        miyagi = total // 2
        sendai = total - miyagi
        out[bucket] = {"miyagi": miyagi, "sendai": sendai}
    return out


def _filter_eligible(entries: Iterable[TocEntry]) -> list[TocEntry]:
    """Keep only ``kind == part`` entries with both a part_number and a Miyagi/Sendai POV."""
    return [
        e
        for e in entries
        if e.kind == "part"
        and e.part_number is not None
        and e.pov in ("miyagi", "sendai")
    ]


def build_holdout(
    *,
    target_count: int | None = None,
    seed: int | None = None,
    lookup: POVLookup | None = None,
) -> HoldoutPlan:
    """Select a stratified holdout. Deterministic given ``seed``."""
    target_count = target_count or THRESHOLDS.test_holdout_target_count
    seed = seed if seed is not None else THRESHOLDS.random_seed
    lookup = lookup or load_pov_lookup()
    rng = random.Random(seed)

    eligible = _filter_eligible(lookup.entries.values())
    total = max((e.part_number for e in eligible if e.part_number is not None), default=0)

    by_cell: dict[tuple[str, POV], list[TocEntry]] = defaultdict(list)
    for entry in eligible:
        bucket = _bucket_for(entry.part_number, total)  # type: ignore[arg-type]
        by_cell[(bucket, entry.pov)].append(entry)

    allocations = _allocate(target_count)
    selected: list[TocEntry] = []
    per_bucket_counts: dict[str, dict[str, int]] = {}
    for bucket, pov_quota in allocations.items():
        per_bucket_counts[bucket] = {}
        for pov, quota in pov_quota.items():
            cell = list(by_cell.get((bucket, pov), []))
            rng.shuffle(cell)
            picked = cell[:quota]
            selected.extend(picked)
            per_bucket_counts[bucket][pov] = len(picked)

    selected.sort(key=lambda e: e.part_number or 0)
    return HoldoutPlan(
        part_ids=[e.id for e in selected],
        seed=seed,
        target_count=target_count,
        per_bucket_counts=per_bucket_counts,
    )


def write_holdout(plan: HoldoutPlan) -> None:
    PATHS.metadata.mkdir(parents=True, exist_ok=True)
    PATHS.holdout_json.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_holdout() -> HoldoutPlan | None:
    if not PATHS.holdout_json.exists():
        return None
    raw = json.loads(PATHS.holdout_json.read_text(encoding="utf-8"))
    return HoldoutPlan(
        part_ids=list(raw.get("part_ids", [])),
        seed=int(raw.get("seed", THRESHOLDS.random_seed)),
        target_count=int(raw.get("target_count", THRESHOLDS.test_holdout_target_count)),
        per_bucket_counts=raw.get("per_bucket_counts", {}),
    )
