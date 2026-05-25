"""Smoke tests for Kakuyomu HTML parsing — both work index and episode body.

These exercise the paragraph kind classifier that drives the prompt's
structured rendering:

  * ``<p class="blank">`` → ``kind="blank"`` (preserved as empty line in
    the ``.txt`` rendering so scene breaks survive into the prompt)
  * leading 「 → ``kind="dialogue"`` (used by the dialogue-parity validator)
  * ruby annotations are stripped down to the kanji
"""

from __future__ import annotations

from translator.scraper.kakuyomu import parse_episode, parse_work_chapters

WORK_HTML_SAMPLE = """
<html>
<head>
<script id="__NEXT_DATA__" type="application/json">
{"props":{"pageProps":{"__APOLLO_STATE__":{
  "Work:1177354054894027232":{
    "tableOfContentsV2":[
      {"__ref":"TableOfContentsChapter:c1"}
    ]
  },
  "TableOfContentsChapter:c1":{
    "chapter":{"__ref":"Chapter:c1"},
    "episodeUnions":[
      {"__ref":"Episode:e1"},
      {"__ref":"Episode:e2"}
    ]
  },
  "Chapter:c1":{"id":"c1","level":1,"title":"テスト章"},
  "Episode:e1":{"id":"e1","title":"第1話"},
  "Episode:e2":{"id":"e2","title":"第2話"}
}}}}
</script>
</head><body></body></html>
"""

EPISODE_HTML_SAMPLE = """
<html><body>
<div class="widget-episodeBody js-episode-body">
  <p id="p1">　別に、<ruby><rb>仙台</rb><rp>(</rp><rt>せんだい</rt><rp>)</rp></ruby>さんでなければならない。</p>
  <p class="blank" id="p2"><br/></p>
  <p id="p3">　次の段落。</p>
  <p id="p4">「<ruby><rb>宮城</rb><rp>(</rp><rt>みやぎ</rt><rp>)</rp></ruby>、これを取って」</p>
  <p class="blank" id="p5"><br/></p>
  <p id="p6">　最後の段落。</p>
</div>
</body></html>
"""


def test_parse_work_chapters_returns_ordered_episode_tree():
    chapters = parse_work_chapters(WORK_HTML_SAMPLE)
    assert len(chapters) == 1
    c = chapters[0]
    assert c.id == "c1"
    assert c.title == "テスト章"
    assert c.episode_ids == ["e1", "e2"]
    assert c.episode_titles == ["第1話", "第2話"]


def test_parse_episode_classifies_paragraphs_and_strips_ruby():
    pc = parse_episode(
        EPISODE_HTML_SAMPLE,
        episode_id="e1",
        entry_id="part_001",
        url="https://kakuyomu.jp/works/X/episodes/e1",
    )
    assert pc.side == "jp"
    assert pc.id == "part_001"

    kinds = [p.kind for p in pc.paragraphs]
    assert kinds == ["narration", "blank", "narration", "dialogue", "blank", "narration"]

    # Ruby annotations should be replaced by kanji only (no furigana left behind).
    full_text = " ".join(p.text for p in pc.paragraphs if p.text)
    assert "仙台" in full_text
    assert "宮城" in full_text
    assert "せんだい" not in full_text
    assert "みやぎ" not in full_text

    # Blank paragraphs should carry empty text so the .txt rendering
    # preserves them as empty lines (the prompt's scene-break signal).
    blanks = [p for p in pc.paragraphs if p.kind == "blank"]
    assert all(p.text == "" for p in blanks)
    assert len(blanks) == 2

    # Dialogue paragraph starts with the JP corner bracket.
    dlg = next(p for p in pc.paragraphs if p.kind == "dialogue")
    assert dlg.text.lstrip("\u3000 \t").startswith("「")
