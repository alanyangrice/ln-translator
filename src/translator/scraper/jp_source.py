"""Bulk JP (Kakuyomu) episode downloader driven by ``toc.json``."""

from __future__ import annotations

import json

from tqdm import tqdm

from translator.config import PATHS
from translator.scraper.kakuyomu import fetch_episode
from translator.scraper.models import PartContent, TocEntry, is_translation_target
from translator.scraper.persist import write_part_content


def load_toc() -> list[TocEntry]:
    if not PATHS.toc_json.exists():
        raise RuntimeError(
            "data/metadata/toc.json missing. Run `translator scrape toc` first."
        )
    raw = json.loads(PATHS.toc_json.read_text(encoding="utf-8"))
    return [TocEntry.model_validate(row) for row in raw]


def scrape_all_jp(
    *,
    only_ids: set[str] | None = None,
    limit: int | None = None,
    refresh: bool = False,
) -> dict[str, int]:
    """Bulk-fetch JP episodes for the numbered ``part`` entries.

    Defaults to v3 pipeline scope (numbered miyagi/sendai parts only).
    Pass an explicit ``only_ids`` set to fetch interludes / extras /
    side stories on a one-off basis; the ``--only`` CLI flag forwards
    here.
    """
    entries = load_toc()
    if only_ids:
        entries = [e for e in entries if e.id in only_ids]
    else:
        entries = [e for e in entries if is_translation_target(e)]
    if limit is not None:
        entries = entries[:limit]

    stats = {"attempted": 0, "succeeded": 0, "skipped_no_episode": 0, "errors": 0}
    for entry in tqdm(entries, desc="JP scrape"):
        stats["attempted"] += 1
        if not entry.kakuyomu_episode_id:
            stats["skipped_no_episode"] += 1
            continue
        try:
            content: PartContent = fetch_episode(
                entry.kakuyomu_episode_id, entry_id=entry.id, refresh=refresh
            )
            write_part_content(content)
            stats["succeeded"] += 1
        except Exception as exc:
            stats["errors"] += 1
            tqdm.write(f"[error] {entry.id}: {exc!r}")
    return stats
