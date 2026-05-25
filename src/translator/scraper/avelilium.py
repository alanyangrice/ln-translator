"""Ave Lilium EN translation scraping: ToC page + per-post body.

The ToC at https://avelilium.com/story-about-buying-my-classmate-once-a-week/ is
a flat WordPress post that lists every translated unit with its POV labeled in
the chapter heading. Posts live on two hosts: older entries on
``amawashigroup.wordpress.com`` (the translator's previous host) and newer
entries on ``avelilium.com``. Both use the same WordPress ``entry-content``
container, so a single post parser works for both.

Per-paragraph kind classification:

  * leading "「" (or English curly-quote double-dialogue) → ``dialogue``
  * otherwise → ``narration``

The EN translator collapses scene-break blank lines (unlike Kakuyomu), so no
``blank`` paragraphs are emitted on the EN side. JP blank markers remain the
canonical scene boundary signal in the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag

from translator.config import SCRAPER
from translator.scraper.http_client import fetch
from translator.scraper.models import POV, Paragraph, PartContent


@dataclass
class TocLink:
    """A raw link extracted from the avelilium ToC page (pre-mapping)."""

    label: str            # e.g. "Part 1", "Interlude", "Extra Part", "Side Story 1 (read after part 15)"
    href: str
    chapter_block: str    # the full chapter heading line this link appeared under
    block_index: int      # ordinal position of the chapter block on the page


# ---------------------------------------------------------------------------
# ToC parsing
# ---------------------------------------------------------------------------

_VOLUME_BREAK_TEXT = re.compile(
    r"This is where volume (\w+) of the light novel ends", re.IGNORECASE
)


def fetch_toc_html(*, refresh: bool = False) -> str:
    res = fetch(SCRAPER.avelilium_toc_url, bucket="toc", refresh=refresh)
    return res.text


def _toc_root(html: str) -> Tag:
    soup = BeautifulSoup(html, "lxml")
    body = soup.find(class_="entry-content") or soup.find("article") or soup
    return body


def parse_toc_links(html: str) -> tuple[list[TocLink], list[int]]:
    """Return (links_in_order, volume_break_block_indices).

    A "block" here is a single ``<p>`` element on the ToC page. The translator
    puts each chapter (with its part links) into one ``<p>``.
    """
    body = _toc_root(html)
    links: list[TocLink] = []
    volume_breaks: list[int] = []

    for block_idx, p in enumerate(body.find_all(["p", "h2", "h3"])):
        block_text = p.get_text(" ", strip=True)
        if _VOLUME_BREAK_TEXT.search(block_text):
            volume_breaks.append(block_idx)
            continue
        for a in p.find_all("a", href=True):
            label = a.get_text(strip=True)
            href = a["href"]
            if not href.startswith("http"):
                continue
            # Filter obviously non-story links (kakuyomu, twitter, amazon, etc).
            if not (
                "amawashigroup.wordpress.com" in href
                or "avelilium.com" in href
            ):
                continue
            # Skip the "find here (you will be taken to Kakuyomu)" callout etc.
            if "kakuyomu" in href:
                continue
            links.append(
                TocLink(label=label, href=href, chapter_block=block_text, block_index=block_idx)
            )
    return links, volume_breaks


# ---------------------------------------------------------------------------
# ToC link → typed entry classification
# ---------------------------------------------------------------------------

_POV_TAG_RE = re.compile(
    r"\((?P<pov>Miyagi|Sendai|Maika|Both)\s*PoVs?\)",
    re.IGNORECASE,
)

_CHAPTER_NUMBER_RE = re.compile(
    r"Chapter\s+([A-Za-z\-]+)",  # e.g. "Chapter Sixty-eight"
    re.IGNORECASE,
)

_NUMBERED_PART_LABEL_RE = re.compile(r"^Part\s+(\d+)$", re.IGNORECASE)
_INTERLUDE_LABEL_RE = re.compile(r"interlude", re.IGNORECASE)
_EXTRA_LABEL_RE = re.compile(r"^extra(\s+part)?$|^bookwalker", re.IGNORECASE)
_MAIKA_SS_LABEL_RE = re.compile(r"side story", re.IGNORECASE)
_BOOKWALKER_LABEL_RE = re.compile(r"bookwalker", re.IGNORECASE)
_READ_AFTER_RE = re.compile(r"read after part\s+(\d+)", re.IGNORECASE)
_SIDE_STORY_VOL_RE = re.compile(r"Maika Side Story Vol\.\s*(\d+)", re.IGNORECASE)
_INTERLUDE_VOL_RE = re.compile(r"Volume\s+([A-Za-z]+)\s+Interlude", re.IGNORECASE)
_EXTRA_VOL_RE = re.compile(r"Volume\s+([A-Za-z]+)\s+Extra", re.IGNORECASE)
_BOOKWALKER_VOL_RE = re.compile(r"Volume\s+([A-Za-z]+)\s+Side Story", re.IGNORECASE)

_EN_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}


def _word_to_number(word: str) -> int | None:
    word = word.lower().replace(" ", "").replace("-", "")
    if word.isdigit():
        return int(word)
    if word in _EN_NUMBER_WORDS:
        return _EN_NUMBER_WORDS[word]
    # Compound forms like "thirty-eight" → "thirtyeight"
    for tens in ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"):
        if word.startswith(tens):
            rest = word[len(tens) :]
            if not rest:
                return _EN_NUMBER_WORDS[tens]
            if rest in _EN_NUMBER_WORDS:
                return _EN_NUMBER_WORDS[tens] + _EN_NUMBER_WORDS[rest]
    return None


def _pov_from_block(block: str) -> POV:
    m = _POV_TAG_RE.search(block)
    if not m:
        return "miyagi"  # safest default; toc.json review will catch any miss
    tag = m.group("pov").lower()
    if tag == "miyagi":
        return "miyagi"
    if tag == "sendai":
        return "sendai"
    if tag == "maika":
        return "maika"
    return "both"


def _chapter_number_from_block(block: str) -> int | None:
    m = _CHAPTER_NUMBER_RE.search(block)
    if not m:
        return None
    return _word_to_number(m.group(1))


def _chapter_title_from_block(block: str) -> str | None:
    """Strip the leading "Chapter X - " prefix and trailing "(POV)" tag.

    The real avelilium ToC uses a U+2013 (EN DASH) between the chapter number
    and the title; the regex matches both that and a plain hyphen-minus.
    """
    block = re.sub(r"^Chapter\s+[A-Za-z\-]+\s*[\u2013\-]\s*", "", block)
    block = _POV_TAG_RE.sub("", block).strip()
    # Drop the "[Part N]" tails which are link labels for parts on this same line.
    block = re.sub(r"\[(Part|Interlude|Extra|Side Story).*", "", block).strip()
    return block.rstrip("\u2013-").strip() or None


# ---------------------------------------------------------------------------
# Post-body parsing
# ---------------------------------------------------------------------------

_DIALOGUE_OPENERS = ("「", "『", "“", '"')


def _classify_paragraph(text: str) -> str:
    stripped = text.lstrip().lstrip("\u00a0").lstrip()
    if not stripped:
        return "blank"
    if stripped.startswith(_DIALOGUE_OPENERS):
        return "dialogue"
    return "narration"


_LEADING_INDENT_RE = re.compile(r"^[\xa0\s]+")
_TRAILING_WS_RE = re.compile(r"[\xa0\s]+$")
_NAV_LINE_RE = re.compile(
    r"(previous part|next part|previous chapter|next chapter)",
    re.IGNORECASE,
)


def parse_post(html: str, *, entry_id: str, url: str) -> PartContent:
    soup = BeautifulSoup(html, "lxml")
    body = soup.find(class_="entry-content") or soup.find("article")
    if body is None:
        raise RuntimeError(f"No entry-content on EN post {url}")

    # Real post title from og:title; fall back to h1.entry-title.
    title_native: str | None = None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title_native = og["content"]
    else:
        h1 = soup.find("h1", class_="entry-title")
        if h1:
            title_native = h1.get_text(strip=True)

    paragraphs: list[Paragraph] = []
    for p in body.find_all("p"):
        # Replace explicit <br> with newlines, then strip.
        for br in p.find_all("br"):
            br.replace_with("\n")
        raw = p.get_text()
        text = _LEADING_INDENT_RE.sub("", raw)
        text = _TRAILING_WS_RE.sub("", text)
        text = text.replace("\u00a0", " ")
        if not text:
            continue  # ignore truly empty paragraphs
        if _NAV_LINE_RE.search(text):
            continue  # strip "< Previous Part | Next Part >" navigation
        kind = _classify_paragraph(text)
        paragraphs.append(Paragraph(index=len(paragraphs), kind=kind, text=text))

    return PartContent(
        id=entry_id,
        side="en",
        source_url=url,
        title_native=title_native,
        paragraphs=paragraphs,
    )


def fetch_post(url: str, *, entry_id: str, refresh: bool = False) -> PartContent:
    res = fetch(url, bucket="en", refresh=refresh)
    return parse_post(res.text, entry_id=entry_id, url=url)
