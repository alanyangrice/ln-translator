"""Bulk EN (avelilium/amawashigroup) post downloader driven by ``toc.json``."""

from __future__ import annotations

from tqdm import tqdm

from translator.scraper.avelilium import fetch_post
from translator.scraper.jp_source import load_toc
from translator.scraper.models import is_translation_target
from translator.scraper.persist import write_part_content


def scrape_all_en(
    *,
    only_ids: set[str] | None = None,
    limit: int | None = None,
    refresh: bool = False,
) -> dict[str, int]:
    """Bulk-fetch EN posts for the numbered ``part`` entries.

    Defaults to v3 pipeline scope (numbered miyagi/sendai parts only).
    Pass an explicit ``only_ids`` set to fetch interludes / extras /
    side stories on a one-off basis.
    """
    entries = load_toc()
    if only_ids:
        entries = [e for e in entries if e.id in only_ids]
    else:
        entries = [e for e in entries if is_translation_target(e)]
    if limit is not None:
        entries = entries[:limit]

    stats = {"attempted": 0, "succeeded": 0, "skipped_no_url": 0, "errors": 0}
    for entry in tqdm(entries, desc="EN scrape"):
        stats["attempted"] += 1
        if not entry.url_en:
            stats["skipped_no_url"] += 1
            continue
        try:
            content = fetch_post(entry.url_en, entry_id=entry.id, refresh=refresh)
            write_part_content(content)
            stats["succeeded"] += 1
        except Exception as exc:
            stats["errors"] += 1
            tqdm.write(f"[error] {entry.id}: {exc!r}")
    return stats
