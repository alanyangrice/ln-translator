"""Tests for Ave Lilium EN post parsing and ToC link classification.

Focuses on the per-paragraph classification (dialogue vs narration), navigation
chrome stripping, and POV inference — especially the Vol. 6 Extra split-POV
case where the chapter heading and the Extra's own label carry different PoV
tags within the same block.
"""

from __future__ import annotations

import pytest

from translator.scraper.avelilium import TocLink, parse_post
from translator.scraper.toc import _classify_link, _pov_for_entry

POST_HTML_SAMPLE = """
<html><head>
<meta property="og:title" content="[Part 1] Test Title (I)">
</head><body>
<article>
<div class="entry-content">
  <p>&nbsp;&nbsp;&nbsp;&nbsp;Plain narration paragraph one.</p>
  <p>&nbsp;&nbsp;&nbsp;&nbsp;「Dialogue line in Japanese brackets.」</p>
  <p>&nbsp;&nbsp;&nbsp;&nbsp;Another narration line.</p>
  <p>&nbsp;</p>
  <p>< Previous Part | Next Part ></p>
</div>
</article>
</body></html>
"""


def test_parse_post_strips_chrome_and_classifies():
    pc = parse_post(POST_HTML_SAMPLE, entry_id="part_001", url="https://example.com/p1")
    assert pc.side == "en"
    assert pc.title_native == "[Part 1] Test Title (I)"
    # Navigation chrome and empty paragraph should be dropped.
    kinds = [p.kind for p in pc.paragraphs]
    assert kinds == ["narration", "dialogue", "narration"]
    assert pc.paragraphs[0].text.startswith("Plain narration")
    assert pc.paragraphs[1].text.startswith("「Dialogue")
    # Indentation (`&nbsp;`) should be stripped from the start of each paragraph.
    assert not pc.paragraphs[0].text.startswith(" ")


@pytest.mark.parametrize(
    ("label", "block", "expected"),
    [
        ("Part 1", "Chapter One – Foo (Miyagi PoV) Part 1 Part 2", "part"),
        ("Interlude", "Volume One Interlude (Chapter 5.5) – Bar (Sendai PoV) Interlude", "interlude"),
        ("Extra Part", "Volume One Extra – Baz (Miyagi PoV) Extra Part", "extra"),
        ("Side Story 1 (read after part 15)", "Maika Side Story Vol. 1 (optional) – Qux Side Story 1 (read after part 15)", "side_story_maika"),
        ("Bookwalker Exclusive Side Story", "Volume Two Side Story (Bookwalker Exclusive) – Foo Bookwalker Exclusive Side Story", "bookwalker"),
        ("Part 1", "April Fool's Fanfiction (Maika x Umina) Part 1", "special"),
    ],
)
def test_classify_link(label, block, expected):
    assert _classify_link(TocLink(label=label, href="https://x", chapter_block=block, block_index=0)) == expected


def test_pov_for_entry_handles_vol6_extra_both_povs():
    """Vol. 6 Extra block contains BOTH a chapter-level (Sendai PoV) tag and
    its own (Both PoVs) tag. The Extra's POV must come from the latter."""
    block = (
        "Chapter Fifty-eight – I'm Too Soft on Miyagi (Sendai PoV) "
        "Part 170 Part 171 "
        "Volume Six Extra (Chapter 58.5) – The Morning After We Crossed the Line as Roommates (Both PoVs) [Extra]"
    )
    assert _pov_for_entry("extra", block, "Extra") == "both"
    # The Part links from the same block must still resolve to the chapter's PoV.
    assert _pov_for_entry("part", block, "Part 170") == "sendai"


def test_pov_for_entry_defaults_maika_for_side_stories():
    """Maika Side Story blocks don't carry an explicit PoV tag; we default to maika."""
    block = "Maika Side Story Vol. 1 (optional) – Utsunomiya Maika's Thoughts Side Story 1 (read after part 15)"
    assert _pov_for_entry("side_story_maika", block, "Side Story 1 (read after part 15)") == "maika"
