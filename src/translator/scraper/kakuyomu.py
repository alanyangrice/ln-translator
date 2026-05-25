"""Kakuyomu scraping: work-index (chapters + episode tree) and per-episode body.

Work page (https://kakuyomu.jp/works/{work_id}) embeds the full Apollo state in
``__NEXT_DATA__``. We pull the ordered chapter -> episode tree from there.

Episode pages (https://kakuyomu.jp/works/{work_id}/episodes/{episode_id}) serve
old-school HTML with the body in ``div.widget-episodeBody``. Paragraphs are
``<p>`` elements; scene-break blank lines are ``<p class="blank">``. Furigana is
``<ruby><rb>kanji</rb><rp>(</rp><rt>kana</rt><rp>)</rp></ruby>`` — we strip the
``rt`` so plain text contains only the kanji.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag

from translator.config import SCRAPER
from translator.scraper.http_client import fetch
from translator.scraper.models import Paragraph, PartContent


@dataclass
class KakuyomuChapter:
    """One Kakuyomu chapter (a group of episodes)."""

    id: str
    title: str
    level: int
    episode_ids: list[str]
    episode_titles: list[str]


def work_url() -> str:
    return f"{SCRAPER.kakuyomu_base}/works/{SCRAPER.kakuyomu_work_id}"


def episode_url(episode_id: str) -> str:
    return f"{SCRAPER.kakuyomu_base}/works/{SCRAPER.kakuyomu_work_id}/episodes/{episode_id}"


_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.S,
)


def _extract_next_data(html: str) -> dict:
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise RuntimeError("No __NEXT_DATA__ block on Kakuyomu work page")
    return json.loads(m.group(1))


def parse_work_chapters(html: str) -> list[KakuyomuChapter]:
    """Return the ordered chapter list with each chapter's episode ids/titles."""
    data = _extract_next_data(html)
    apollo = data["props"]["pageProps"]["__APOLLO_STATE__"]
    work_key = f"Work:{SCRAPER.kakuyomu_work_id}"
    work = apollo[work_key]
    toc_refs = work["tableOfContentsV2"]

    chapters: list[KakuyomuChapter] = []
    for toc_ref in toc_refs:
        toc_chapter = apollo[toc_ref["__ref"]]
        chapter_ref = toc_chapter["chapter"]["__ref"]
        chapter = apollo[chapter_ref]
        ep_ids: list[str] = []
        ep_titles: list[str] = []
        for ep_ref in toc_chapter["episodeUnions"]:
            ep = apollo[ep_ref["__ref"]]
            ep_ids.append(ep["id"])
            ep_titles.append(ep.get("title", ""))
        chapters.append(
            KakuyomuChapter(
                id=chapter["id"],
                title=chapter["title"],
                level=chapter.get("level", 1),
                episode_ids=ep_ids,
                episode_titles=ep_titles,
            )
        )
    return chapters


def fetch_work_chapters(*, refresh: bool = False) -> list[KakuyomuChapter]:
    res = fetch(work_url(), bucket="toc", refresh=refresh)
    return parse_work_chapters(res.text)


# ---------------------------------------------------------------------------
# Episode body extraction
# ---------------------------------------------------------------------------

_RUBY_KEEP_RB = True  # keep <rb> kanji, drop <rt> furigana


def _strip_ruby(tag: Tag) -> None:
    """Replace ``<ruby>`` with its ``<rb>`` content; drop ``<rt>`` and ``<rp>``."""
    for ruby in tag.find_all("ruby"):
        rb_text_parts: list[str] = []
        for child in ruby.children:
            if isinstance(child, Tag):
                if child.name == "rb":
                    rb_text_parts.append(child.get_text())
                elif child.name in ("rt", "rp"):
                    continue
                else:
                    rb_text_parts.append(child.get_text())
            else:
                rb_text_parts.append(str(child))
        ruby.replace_with("".join(rb_text_parts))


def _classify_paragraph(text: str, css_classes: list[str]) -> str:
    if "blank" in css_classes:
        return "blank"
    stripped = text.lstrip("\u3000 \t")
    if stripped.startswith("「") or stripped.startswith("『"):
        return "dialogue"
    return "narration"


def parse_episode(html: str, *, episode_id: str, entry_id: str, url: str) -> PartContent:
    """Extract the structured paragraph list from a Kakuyomu episode HTML page."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.find(class_="widget-episodeBody")
    if body is None:
        raise RuntimeError(f"No widget-episodeBody on episode {episode_id}")

    # Episode title from page header.
    title_el = soup.find(class_="widget-episodeTitle") or soup.find("h1")
    title_native = title_el.get_text(strip=True) if title_el else None

    # Normalize ruby annotations in-place: keep kanji, drop furigana.
    _strip_ruby(body)

    paragraphs: list[Paragraph] = []
    for idx, p in enumerate(body.find_all("p")):
        classes = p.get("class") or []
        # Replace explicit <br> with newlines, then collapse whitespace per paragraph.
        for br in p.find_all("br"):
            br.replace_with("\n")
        raw = p.get_text()
        # Kakuyomu indents narration paragraphs with U+3000 (ideographic space).
        # Preserve it: the indent is part of the source text the model sees in
        # the prompt, and matches the JP-side rendering of the reference parts.
        text = raw.rstrip("\n").rstrip()
        kind = _classify_paragraph(text, classes)
        if kind == "blank":
            text = ""
        paragraphs.append(Paragraph(index=idx, kind=kind, text=text))

    return PartContent(
        id=entry_id,
        side="jp",
        source_url=url,
        title_native=title_native,
        paragraphs=paragraphs,
    )


def fetch_episode(episode_id: str, *, entry_id: str, refresh: bool = False) -> PartContent:
    url = episode_url(episode_id)
    res = fetch(url, bucket="jp", refresh=refresh)
    return parse_episode(res.text, episode_id=episode_id, entry_id=entry_id, url=url)
