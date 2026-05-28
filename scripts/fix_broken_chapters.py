"""Re-extract chapters whose paragraph_pairs failed quality gates.

Forces a fresh LLM extraction (with the strengthened prompt + post-parse
validator) on the 23 chapters identified by ``audit_index_v2.py``.
After re-extraction, rebuilds the full index so the new chapter outputs
land in ``phrases/`` and ``paragraphs/``.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from translator.precedents.extract import (
    DEFAULT_EXTRACTION_MODEL,
    default_v2_root,
    extract_chapter,
    extract_corpus,
)

# Chapters flagged by the audit. Keep in sync if more issues surface.
BROKEN_CHAPTERS = [
    "part_014", "part_028", "part_048", "part_051", "part_056",
    "part_059", "part_073", "part_074", "part_081", "part_086",
    "part_094", "part_103", "part_108", "part_114", "part_126",
    "part_138", "part_150", "part_169", "part_191", "part_196",
    "part_210", "part_225", "part_229",
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-pro"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else len(BROKEN_CHAPTERS)

    print(f"=== fix_broken_chapters ===")
    print(f"model:    {model}")
    print(f"workers:  {workers}")
    print(f"chapters: {len(BROKEN_CHAPTERS)}")
    print(f"started:  {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print()

    root = default_v2_root()

    # Phase 1: force re-extract just the broken ones.
    t0 = time.time()
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    def run_one(pid: str):
        ts = time.time()
        try:
            ex = extract_chapter(pid, model=model, cache_root=root, force=True)
            return pid, ex, time.time() - ts, None
        except Exception as exc:  # pylint: disable=broad-except
            return pid, None, time.time() - ts, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, pid): pid for pid in BROKEN_CHAPTERS}
        completed = 0
        for fut in as_completed(futs):
            pid, ex, elapsed, exc = fut.result()
            completed += 1
            if exc is not None:
                failed.append((pid, f"{type(exc).__name__}: {exc}"))
                print(
                    f"[{completed}/{len(BROKEN_CHAPTERS)}] {pid}: "
                    f"FAILED in {elapsed:.1f}s — {type(exc).__name__}: {exc}"
                )
            else:
                succeeded.append(pid)
                print(
                    f"[{completed}/{len(BROKEN_CHAPTERS)}] {pid}: "
                    f"{len(ex.phrase_pairs)} phrase + "
                    f"{len(ex.paragraph_pairs)} paragraph pair(s) "
                    f"({elapsed:.1f}s)"
                )

    extract_elapsed = time.time() - t0
    print()
    print(f"Re-extraction done in {extract_elapsed:.1f}s")
    print(f"  succeeded: {len(succeeded)}/{len(BROKEN_CHAPTERS)}")
    print(f"  failed:    {len(failed)}")
    if failed:
        for pid, err in failed:
            print(f"    {pid}: {err}")

    if failed:
        print("\n⚠️  Skipping index rebuild because some chapters failed.")
        return 1

    # Phase 2: rebuild the global index by running extract_corpus over
    # ALL chapters (cached entries are reused, so this is cheap and
    # ensures the regenerated chapters land in pairs.jsonl /
    # embeddings.npy alongside the others).
    print("\n=== rebuilding global index ===")
    t1 = time.time()
    stats = extract_corpus(
        parts=None,
        rebuild=False,
        extraction_model=DEFAULT_EXTRACTION_MODEL,  # only for cache misses
        progress=False,
        workers=64,
    )
    print(
        f"  rebuilt in {time.time() - t1:.1f}s — "
        f"{stats.phrase_count} phrases, {stats.paragraph_count} paragraphs"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
