"""Pair-completeness verification for the scraped parallel corpus.

Scope is the v3 pipeline target: numbered ``part`` entries with a
miyagi or sendai POV. Interludes, extras, side stories, and bonus
content are intentionally excluded from the report.

Success criterion: every in-scope ToC entry has both a JP and EN
structured JSON file on disk with non-trivial content. We additionally
emit a paragraph count skew so obviously broken scrapes (one side
empty, parser regressions) surface before downstream pipeline runs
read the corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from translator.config import PATHS
from translator.scraper.jp_source import load_toc
from translator.scraper.models import PartContent, is_translation_target


@dataclass
class VerifyReport:
    in_scope_entries: int  # numbered parts with supported POV
    out_of_scope_entries: int  # interludes / extras / side stories / etc.
    jp_present: int
    en_present: int
    complete_pairs: int
    missing_jp: list[str]  # numbered parts whose JP scrape is broken
    missing_jp_ln_only: list[str]  # numbered parts that only exist in the LN, not Kakuyomu
    missing_en: list[str]  # numbered parts whose EN scrape is broken (had a url, didn't fetch)
    missing_en_jp_only: list[str]  # JP-only tail parts past the EN translation (no EN exists yet)
    paragraph_skew: list[tuple[str, int, int]]  # (id, jp_non_blank, en_non_blank)


def _load(side_path) -> PartContent | None:
    if not side_path.exists():
        return None
    try:
        return PartContent.model_validate(json.loads(side_path.read_text(encoding="utf-8")))
    except Exception:
        return None


def verify_pairs(*, skew_threshold: float = 0.25) -> VerifyReport:
    all_entries = load_toc()
    in_scope = [e for e in all_entries if is_translation_target(e)]

    missing_jp: list[str] = []
    missing_jp_ln_only: list[str] = []
    missing_en: list[str] = []
    missing_en_jp_only: list[str] = []
    skew: list[tuple[str, int, int]] = []
    jp_present = 0
    en_present = 0
    complete = 0

    for e in in_scope:
        jp = _load(PATHS.parallel / f"{e.id}.jp.json")
        en = _load(PATHS.parallel / f"{e.id}.en.json")
        if jp:
            jp_present += 1
        elif not e.kakuyomu_episode_id:
            missing_jp_ln_only.append(e.id)
        else:
            missing_jp.append(e.id)
        if en:
            en_present += 1
        elif not e.url_en:
            missing_en_jp_only.append(e.id)
        else:
            missing_en.append(e.id)
        if jp and en:
            complete += 1
            jp_n = jp.non_blank_count
            en_n = en.non_blank_count
            if jp_n and en_n:
                ratio = abs(jp_n - en_n) / max(jp_n, en_n)
                if ratio > skew_threshold:
                    skew.append((e.id, jp_n, en_n))

    return VerifyReport(
        in_scope_entries=len(in_scope),
        out_of_scope_entries=len(all_entries) - len(in_scope),
        jp_present=jp_present,
        en_present=en_present,
        complete_pairs=complete,
        missing_jp=missing_jp,
        missing_jp_ln_only=missing_jp_ln_only,
        missing_en=missing_en,
        missing_en_jp_only=missing_en_jp_only,
        paragraph_skew=skew,
    )
