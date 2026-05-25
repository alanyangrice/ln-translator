"""Tests for the stratified holdout selector.

The selector was previously buggy: it sampled across all 417 Kakuyomu episodes,
so parts in the untranslated tail (>= part_230) ended up in the holdout even
though they have no EN reference for COMET / BERTScore / LLM-judge to score
against. The fix restricts eligibility to parts that actually have an EN
file on disk, and the tests pin that contract.
"""

from __future__ import annotations

from collections import Counter

import pytest

from translator.config import PATHS, THRESHOLDS
from translator.prep.holdout import (
    _bucket_for,
    build_holdout,
    load_holdout,
)
from translator.prep.pov import load_pov_lookup


@pytest.fixture(scope="module")
def lookup():
    return load_pov_lookup()


def _has_en_on_disk(part_id: str) -> bool:
    return (PATHS.parallel / f"{part_id}.en.txt").exists()


def test_every_holdout_member_has_en_reference(lookup):
    """A holdout member without an EN reference is dead weight — eval can't score it."""
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    for pid in plan.part_ids:
        assert _has_en_on_disk(pid), (
            f"holdout includes {pid} but data/parallel/{pid}.en.txt is missing"
        )


def test_holdout_only_contains_kind_part(lookup):
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    for pid in plan.part_ids:
        entry = lookup.get(pid)
        assert entry.kind == "part", (
            f"holdout includes {pid} which is a {entry.kind!r} entry; "
            "interludes/extras/SS use separate evaluation paths"
        )


def test_holdout_only_contains_supported_povs(lookup):
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    for pid in plan.part_ids:
        entry = lookup.get(pid)
        assert entry.pov in {"miyagi", "sendai"}, (
            f"holdout includes {pid} with POV {entry.pov!r}; Maika is handled manually"
        )


def test_holdout_is_deterministic_across_calls(lookup):
    a = build_holdout(target_count=30, seed=7, lookup=lookup)
    b = build_holdout(target_count=30, seed=7, lookup=lookup)
    assert a.part_ids == b.part_ids
    assert a.per_bucket_counts == b.per_bucket_counts


def test_holdout_changes_with_seed(lookup):
    a = build_holdout(target_count=30, seed=7, lookup=lookup)
    b = build_holdout(target_count=30, seed=13, lookup=lookup)
    assert a.part_ids != b.part_ids, (
        "different seeds produced identical holdouts — selector is not actually seeded"
    )


def test_holdout_size_matches_target(lookup):
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    assert len(plan.part_ids) == 30


def test_holdout_buckets_are_balanced(lookup):
    """The per-bucket totals should differ by at most 1 (rounding remainder)."""
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    bucket_totals = {
        b: sum(plan.per_bucket_counts.get(b, {}).values()) for b in ("early", "mid", "late")
    }
    spread = max(bucket_totals.values()) - min(bucket_totals.values())
    assert spread <= 1, f"buckets too unbalanced: {bucket_totals}"


def test_holdout_pov_split_is_balanced(lookup):
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    povs = Counter(lookup.get(p).pov for p in plan.part_ids)
    miyagi = povs.get("miyagi", 0)
    sendai = povs.get("sendai", 0)
    assert abs(miyagi - sendai) <= 2, f"POV split unbalanced: {povs}"


def test_holdout_part_numbers_span_full_corpus(lookup):
    """The test set should touch all three story slices, not cluster in one place."""
    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    nums = sorted(lookup.get(p).part_number for p in plan.part_ids)
    # With 30 picks from the 229-part parallel corpus, the smallest pick
    # should be well below part 50 and the largest well above part 180.
    assert nums[0] < 50, f"holdout doesn't reach early parts (min={nums[0]})"
    assert nums[-1] > 180, f"holdout doesn't reach late parts (max={nums[-1]})"


def test_bucket_assignment():
    # With total=229 (parallel corpus size): early < 76.3, mid < 152.7, late >= 152.7
    assert _bucket_for(1, 229) == "early"
    assert _bucket_for(75, 229) == "early"
    assert _bucket_for(100, 229) == "mid"
    assert _bucket_for(150, 229) == "mid"
    assert _bucket_for(160, 229) == "late"
    assert _bucket_for(229, 229) == "late"


def test_load_holdout_roundtrips_what_was_written(lookup):
    """Whatever build → write produces, load returns the same thing."""
    from translator.prep.holdout import write_holdout

    plan = build_holdout(target_count=30, seed=7, lookup=lookup)
    backup = PATHS.holdout_json.read_text(encoding="utf-8") if PATHS.holdout_json.exists() else None
    try:
        write_holdout(plan)
        loaded = load_holdout()
        assert loaded is not None
        assert loaded.part_ids == plan.part_ids
        assert loaded.seed == plan.seed
        assert loaded.target_count == plan.target_count
    finally:
        # Restore prior holdout.json so the test doesn't have side effects.
        if backup is not None:
            PATHS.holdout_json.write_text(backup, encoding="utf-8")


def test_default_thresholds_target_count_is_30():
    """v3 plan locks the target at 30; tests rely on that elsewhere."""
    assert THRESHOLDS.test_holdout_target_count == 30
