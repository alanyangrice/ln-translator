"""Tests for the structural validation checks.

The validators are pure-Python checks that run against synthetic strings —
no filesystem, no network — so they're cheap to exercise across the corner
cases that matter for catching catastrophic translation failures.
"""

from __future__ import annotations

from translator.config import THRESHOLDS
from translator.validation.checks import (
    check_dialogue_parity,
    check_length_ratio,
    check_name_frequency,
    validate_translation,
)

# ---------------------------------------------------------------------------
# Dialogue parity
# ---------------------------------------------------------------------------


def test_dialogue_parity_passes_when_counts_match_exactly():
    jp = "「こんにちは」と仙台さんが言った。「うん」と私は答えた。"
    en = "「Hello,」 Sendai-san said. 「Yeah,」 I answered."
    result = check_dialogue_parity(jp, en)
    assert result.status == "pass"
    assert result.details["jp_blocks"] == result.details["en_blocks"] == 2


def test_dialogue_parity_passes_when_no_dialogue_either_side():
    jp = "宮城は窓の外を眺めていた。雪が降っていた。"
    en = "Miyagi gazed out the window. It was snowing."
    result = check_dialogue_parity(jp, en)
    assert result.status == "pass"


def test_dialogue_parity_fails_when_translation_drops_lines():
    jp = "「" + "」「".join("台詞" + str(i) for i in range(20)) + "」"
    en = "Just one line in English."
    result = check_dialogue_parity(jp, en)
    assert result.status == "fail"
    assert result.details["skew"] > THRESHOLDS.dialogue_parity_max_skew


def test_dialogue_parity_handles_straight_quotes_fallback():
    """When the model switches to straight double quotes instead of 「」."""
    jp = "「Hi」「Bye」"
    en = '"Hi" "Bye"'
    result = check_dialogue_parity(jp, en)
    assert result.status == "pass"
    assert result.details["en_blocks"] == 2


def test_dialogue_parity_handles_curly_quotes_fallback():
    jp = "「Hi」「Bye」"
    en = "\u201cHi\u201d \u201cBye\u201d"
    result = check_dialogue_parity(jp, en)
    assert result.status == "pass"
    assert result.details["en_blocks"] == 2


def test_dialogue_parity_counts_min_of_open_close():
    """An unbalanced 「 without matching 」 should not be counted as a complete block."""
    jp = "「open「open」one_close"
    result = check_dialogue_parity(jp, "")
    assert result.details["jp_blocks"] == 1


# ---------------------------------------------------------------------------
# Name frequency
# ---------------------------------------------------------------------------


def test_name_frequency_passes_when_counts_align():
    jp = "仙台さんは笑った。仙台さんは話した。宮城は答えた。"
    en = "Sendai-san laughed. Sendai-san spoke. Miyagi answered."
    result = check_name_frequency(jp, en)
    assert result.status == "pass"


def test_name_frequency_passes_when_no_names_appear():
    jp = "今日は雨だった。"
    en = "It rained today."
    result = check_name_frequency(jp, en)
    assert result.status == "pass"
    assert result.details["rows"] == []


def test_name_frequency_fails_when_character_vanishes():
    """A catastrophic name-elision: JP has Miyagi everywhere, EN has none."""
    jp = "宮城。" * 20
    en = "She did things. " * 20
    result = check_name_frequency(jp, en)
    assert result.status == "fail"
    assert result.details["worst_skew"] >= THRESHOLDS.name_frequency_fail_skew


def test_name_frequency_warn_band():
    """Asymmetric counts within the warn band → status 'warn'.

    Uses Miyagi (single EN form) so the test isn't muddled by the
    "Sendai" / "Sendai-san" overlap inside the SENDAI matcher.
    """
    # JP=10, EN=3 → skew = 0.7 → between warn (0.65) and fail (0.90).
    jp = "宮城" * 10
    en = "Miyagi did. " * 3
    result = check_name_frequency(jp, en)
    assert result.status == "warn"


def test_name_frequency_counts_both_sendai_forms():
    """EN can spell the name either 'Sendai-san' or 'Sendai'; both count."""
    jp = "仙台" * 4
    en = "Sendai-san Sendai-san Sendai Sendai"  # 4 mentions across both forms
    result = check_name_frequency(jp, en)
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# Length ratio
# ---------------------------------------------------------------------------


def test_length_ratio_passes_at_corpus_mean():
    """Mean ratio in the corpus is ~2.33; aim for the middle of the band."""
    jp = "あ" * 100
    en = "x" * 230
    result = check_length_ratio(jp, en)
    assert result.status == "pass"


def test_length_ratio_fails_when_translation_is_too_short():
    jp = "あ" * 100
    en = "x" * 50  # ratio = 0.5, far below min
    result = check_length_ratio(jp, en)
    assert result.status == "fail"


def test_length_ratio_fails_when_translation_is_too_long():
    jp = "あ" * 100
    en = "x" * 1000  # ratio = 10.0, far above max
    result = check_length_ratio(jp, en)
    assert result.status == "fail"


def test_length_ratio_warns_just_outside_band():
    jp = "あ" * 100
    # length_ratio_max=2.94; warn band extends to hi*1.15 = 3.38
    en = "x" * 300  # ratio = 3.0 → warn
    result = check_length_ratio(jp, en)
    assert result.status == "warn"


def test_length_ratio_ignores_whitespace():
    """Whitespace runs shouldn't pad or shrink the visible-char count."""
    jp = "あ" * 100
    en_no_ws = "x" * 230
    en_padded = "    \n\n".join("x" * 23 for _ in range(10))  # same 230 visible
    a = check_length_ratio(jp, en_no_ws)
    b = check_length_ratio(jp, en_padded)
    assert a.details["en_chars"] == b.details["en_chars"]


def test_length_ratio_warns_on_empty_jp():
    result = check_length_ratio("", "anything")
    assert result.status == "warn"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_validate_translation_returns_all_three_checks():
    report = validate_translation("part_001", "「テスト」", "「test」 said.")
    names = [c.name for c in report.checks]
    assert names == ["dialogue_parity", "name_frequency", "length_ratio"]


def test_report_status_is_worst_check():
    # Construct a translation that fails dialogue but passes the others.
    jp = "「" + "」「".join("x" * 5 for _ in range(20)) + "」"
    en = "a" * (2 * len(jp))  # length ratio ~2; dialogue=0; names=0
    report = validate_translation("p", jp, en)
    assert report.status == "fail"
    assert report.passed is False


def test_report_status_pass_when_all_pass():
    # Tune visible-char counts so the length ratio lands inside the band.
    # JP visible chars = 12, EN visible chars ≈ 28 → ratio ≈ 2.3 (corpus mean).
    jp = "「テストです」と二人は話した。"  # 15 visible JP chars
    en = "「Test,」 the two said quietly together now."  # ~37 visible EN chars
    report = validate_translation("p", jp, en)
    assert report.status == "pass", [
        (c.name, c.status, c.message) for c in report.checks
    ]
    assert report.passed is True
