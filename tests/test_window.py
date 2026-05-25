"""Tests for the sliding-window builder.

The window is the heart of the v3 prompt: get this wrong (wrong order, wrong
exclusions, future leakage) and every translation degrades silently. These
tests exercise the contract against the real corpus on disk.
"""

from __future__ import annotations

import pytest

from translator.config import THRESHOLDS
from translator.inference.window import build_window
from translator.prep.holdout import HoldoutPlan
from translator.prep.pov import load_pov_lookup


@pytest.fixture(scope="module")
def lookup():
    return load_pov_lookup()


def test_window_has_correct_size_in_steady_state(lookup):
    """Once we're past the warmup, the window should be exactly THRESHOLDS.window_size parts."""
    window = build_window("part_050", holdout=None, lookup=lookup)
    assert len(window.parts) == THRESHOLDS.window_size


def test_window_part_numbers_are_strictly_before_target(lookup):
    """No future leakage: every reference part must have a lower part_number."""
    window = build_window("part_050", holdout=None, lookup=lookup)
    for ref in window.parts:
        assert ref.entry.part_number is not None
        assert ref.entry.part_number < 50


def test_window_target_is_never_in_its_own_window(lookup):
    window = build_window("part_050", holdout=None, lookup=lookup)
    assert "part_050" not in [r.entry.id for r in window.parts]


def test_window_is_chronologically_ordered_oldest_first(lookup):
    """v3 places the most-recent reference *last*, just before the new chapter."""
    window = build_window("part_050", holdout=None, lookup=lookup)
    nums = [r.entry.part_number for r in window.parts]
    assert nums == sorted(nums), f"window not in chronological order: {nums}"


def test_window_picks_the_most_recent_eligible_parts(lookup):
    """When the window is full, the last reference should be the part just before the target."""
    window = build_window("part_050", holdout=None, lookup=lookup)
    assert window.parts[-1].entry.id == "part_049"


def test_window_underfills_at_start_and_emits_note(lookup):
    """Translating part_005 has only 4 prior parts; window should partially fill and notes
    should explain why."""
    window = build_window("part_005", holdout=None, lookup=lookup)
    assert len(window.parts) <= 4
    assert any("underfilled" in note for note in window.notes), window.notes


def test_window_for_part_001_is_empty(lookup):
    """No prior parts → empty window plus a note."""
    window = build_window("part_001", holdout=None, lookup=lookup)
    assert window.parts == []
    assert any("underfilled" in note for note in window.notes)


def test_window_skips_holdout_members(lookup):
    """Held-out parts must never appear as in-context examples during eval."""
    # Hold out part_049 — the part immediately before part_050. With it
    # excluded, the window must reach further back.
    holdout = HoldoutPlan(part_ids=["part_049"], seed=0, target_count=1)
    window = build_window("part_050", holdout=holdout, lookup=lookup)
    ids = [r.entry.id for r in window.parts]
    assert "part_049" not in ids
    # The displaced slot should be filled by part_039 reaching one further back.
    assert window.parts[-1].entry.id == "part_048"
    # And we still got a full window because parts_001-048 are translated.
    assert len(window.parts) == THRESHOLDS.window_size


def test_window_includes_parts_in_window_have_both_jp_and_en_text(lookup):
    """Every reference must carry both JP and EN payload; the prompt depends on both."""
    window = build_window("part_050", holdout=None, lookup=lookup)
    for ref in window.parts:
        assert ref.jp_text.strip(), f"empty JP text for {ref.entry.id}"
        assert ref.en_text.strip(), f"empty EN text for {ref.entry.id}"


def test_window_excludes_untranslated_tail(lookup):
    """Translating part_230 (the first untranslated part) should still build a full window
    from the prior translated parts."""
    window = build_window("part_230", holdout=None, lookup=lookup)
    assert len(window.parts) == THRESHOLDS.window_size
    # The most recent reference should be part_229, the last translated one.
    assert window.parts[-1].entry.id == "part_229"


def test_window_skipping_multiple_holdout_neighbors_reaches_further_back(lookup):
    """Stress test: hold out 5 consecutive parts before the target; window backs up by 5."""
    target = "part_100"
    held = [f"part_{n:03d}" for n in range(95, 100)]
    holdout = HoldoutPlan(part_ids=held, seed=0, target_count=len(held))
    window = build_window(target, holdout=holdout, lookup=lookup)
    ids = [r.entry.id for r in window.parts]
    for h in held:
        assert h not in ids
    # The most recent reference should be part_094.
    assert window.parts[-1].entry.id == "part_094"
    assert len(window.parts) == THRESHOLDS.window_size
