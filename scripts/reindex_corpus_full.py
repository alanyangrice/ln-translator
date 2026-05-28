"""Full-corpus reindex with force=True + max parallelism.

Replaces the V4-Flash-built v2 index with a fresh V4-Pro extraction
that runs through the new prompt + paragraph quality gates + verbatim
phrase filter. Logs progress to stdout (tee for replay). Writes
``data/extract_corpus_stats_<model>.json`` at the end.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from translator.precedents.extract import (
    DEFAULT_EXTRACTION_MODEL,
    extract_corpus,
)


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXTRACTION_MODEL
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 229

    print(f"=== reindex_corpus_full ===", flush=True)
    print(f"model:    {model}", flush=True)
    print(f"workers:  {workers}", flush=True)
    print(f"started:  {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(flush=True)

    t0 = time.time()
    stats = extract_corpus(
        parts=None,
        rebuild=True,           # force=True on every chapter
        extraction_model=model,
        progress=True,
        workers=workers,
    )
    elapsed = time.time() - t0

    print(flush=True)
    print(f"=== done in {elapsed:.1f}s ({elapsed/60:.1f} min) ===", flush=True)
    print(f"  processed:       {len(stats.parts_processed)}", flush=True)
    print(f"  failed:          {len(stats.parts_failed)}", flush=True)
    print(f"  phrase_count:    {stats.phrase_count}", flush=True)
    print(f"  paragraph_count: {stats.paragraph_count}", flush=True)
    if stats.parts_failed:
        print(f"  failed parts:    {stats.parts_failed}", flush=True)

    out_path = Path("data") / f"extract_corpus_stats_{model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": model,
                "workers": workers,
                "elapsed_seconds": elapsed,
                "phrase_count": stats.phrase_count,
                "paragraph_count": stats.paragraph_count,
                "parts_processed": sorted(stats.parts_processed),
                "parts_failed": sorted(stats.parts_failed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  stats:           {out_path}", flush=True)
    return 0 if not stats.parts_failed else 1


if __name__ == "__main__":
    sys.exit(main())
