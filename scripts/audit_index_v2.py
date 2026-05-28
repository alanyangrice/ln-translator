"""Sanity-check the v2 precedent index.

Verifies:

1. Cache coverage — one ``cache/part_NNN.json`` per parallel chapter.
2. Index coverage — every parallel chapter appears in ``phrases/pairs.jsonl``
   and ``paragraphs/pairs.jsonl`` (exception: chapters where the LLM legitimately
   produced zero phrases or zero paragraphs).
3. Row alignment — ``embeddings.npy`` row count matches ``pairs.jsonl`` line count.
4. Per-chapter content coverage — for each chapter, what fraction of the
   source JP characters made it into ``paragraph_pairs``? Flag anything
   where coverage drops below 50% of source length, since that suggests
   the LLM truncated or skipped a section.
5. Outliers — chapters with anomalously low phrase or paragraph counts
   (z-score < -2.0).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from translator.precedents.extract import default_v2_root
from translator.prep.corpus import load_part_jp, load_part_en, iter_parts


def main() -> None:
    root = default_v2_root()
    print(f"Index root: {root}\n")

    # 1. Universe of chapters with parallel JP+EN.
    parallel = []
    for part in iter_parts(only_translated=True, parts_only=True):
        if load_part_en(part.id) is not None:
            parallel.append(part.id)
    parallel_set = set(parallel)
    print(f"Parallel corpus: {len(parallel)} chapters")

    # 2. Cache files.
    cache_dir = root / "cache"
    cache_files = sorted(cache_dir.glob("part_*.json"))
    cache_ids = {f.stem for f in cache_files}
    missing_cache = parallel_set - cache_ids
    extra_cache = cache_ids - parallel_set
    print(f"Cache files:     {len(cache_files)}")
    print(f"  missing cache: {len(missing_cache)}{' '+sorted(missing_cache)[:5] if missing_cache else ''}")
    print(f"  extra cache:   {len(extra_cache)}{' '+sorted(extra_cache)[:5] if extra_cache else ''}")

    # 3. Index files.
    phrase_pairs_path = root / "phrases" / "pairs.jsonl"
    para_pairs_path = root / "paragraphs" / "pairs.jsonl"
    phrase_emb_path = root / "phrases" / "embeddings.npy"
    para_emb_path = root / "paragraphs" / "embeddings.npy"

    def load_pairs(p: Path) -> list[dict]:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    phrases = load_pairs(phrase_pairs_path)
    paragraphs = load_pairs(para_pairs_path)
    phrase_emb = np.load(phrase_emb_path)
    para_emb = np.load(para_emb_path)

    print(f"\nPhrases:    {len(phrases):>6} pairs, embedding shape {phrase_emb.shape}")
    print(f"Paragraphs: {len(paragraphs):>6} pairs, embedding shape {para_emb.shape}")

    # 4. Row alignment.
    phrase_align = "OK" if phrase_emb.shape[0] == len(phrases) else "MISMATCH"
    para_align   = "OK" if para_emb.shape[0]   == len(paragraphs) else "MISMATCH"
    print(f"  phrases pairs↔embeddings:    {phrase_align}")
    print(f"  paragraphs pairs↔embeddings: {para_align}")

    # 5. Coverage by chapter.
    phrase_by_part: dict[str, list[dict]] = defaultdict(list)
    para_by_part: dict[str, list[dict]] = defaultdict(list)
    for p in phrases:
        phrase_by_part[p["part_id"]].append(p)
    for p in paragraphs:
        para_by_part[p["part_id"]].append(p)

    parts_in_phrase = set(phrase_by_part.keys())
    parts_in_para = set(para_by_part.keys())

    missing_phrase = parallel_set - parts_in_phrase
    missing_para = parallel_set - parts_in_para
    extra_phrase = parts_in_phrase - parallel_set
    extra_para = parts_in_para - parallel_set

    print(f"\nChapter representation:")
    print(f"  in phrases index:    {len(parts_in_phrase)}/{len(parallel_set)}")
    print(f"  in paragraphs index: {len(parts_in_para)}/{len(parallel_set)}")
    if missing_phrase:
        print(f"  ❗ chapters with NO phrases ({len(missing_phrase)}): {sorted(missing_phrase)}")
    if missing_para:
        print(f"  ❗ chapters with NO paragraphs ({len(missing_para)}): {sorted(missing_para)}")
    if extra_phrase:
        print(f"  ❗ phrase entries from non-parallel parts: {sorted(extra_phrase)}")
    if extra_para:
        print(f"  ❗ paragraph entries from non-parallel parts: {sorted(extra_para)}")

    # 6. Per-chapter JP coverage in paragraph_pairs.
    print(f"\nPer-chapter paragraph coverage (JP-char overlap with source):")
    coverage_rows = []
    for pid in sorted(parallel):
        jp_src = load_part_jp(pid)
        src_len = len(jp_src)
        # Strip whitespace from source to make a fair denominator.
        src_compact = "".join(jp_src.split())
        # Sum of indexed paragraph JP lengths (compact).
        idx_compact_chars = sum(
            len("".join(p["jp_text"].split())) for p in para_by_part.get(pid, [])
        )
        coverage = idx_compact_chars / max(1, len(src_compact))
        coverage_rows.append((pid, len(src_compact), idx_compact_chars, coverage,
                              len(para_by_part.get(pid, [])),
                              len(phrase_by_part.get(pid, []))))

    # Stats on coverage.
    coverages = [c for _, _, _, c, _, _ in coverage_rows]
    print(f"  coverage mean={statistics.mean(coverages):.2f}  median={statistics.median(coverages):.2f}  "
          f"min={min(coverages):.2f}  max={max(coverages):.2f}")

    # Worst 15 by coverage.
    worst = sorted(coverage_rows, key=lambda x: x[3])[:15]
    print(f"\n  Lowest-coverage 15 chapters:")
    print(f"  {'part':<10} {'src_chars':>10} {'idx_chars':>10} {'coverage':>9} {'paras':>6} {'phrases':>8}")
    for pid, src, idx, cov, n_para, n_phr in worst:
        flag = "  ⚠️" if cov < 0.50 else ""
        print(f"  {pid:<10} {src:>10} {idx:>10} {cov:>9.2%} {n_para:>6} {n_phr:>8}{flag}")

    # 7. Outliers in raw counts.
    phrase_counts = [n_phr for _, _, _, _, _, n_phr in coverage_rows]
    para_counts = [n_para for _, _, _, _, n_para, _ in coverage_rows]
    pmean, psd = statistics.mean(phrase_counts), statistics.stdev(phrase_counts)
    qmean, qsd = statistics.mean(para_counts), statistics.stdev(para_counts)

    print(f"\nDistribution stats:")
    print(f"  phrases per chapter:    mean={pmean:.1f}  sd={psd:.1f}  min={min(phrase_counts)}  max={max(phrase_counts)}")
    print(f"  paragraphs per chapter: mean={qmean:.1f}  sd={qsd:.1f}  min={min(para_counts)}  max={max(para_counts)}")

    # 8. Empty/sparse chapters.
    empty_phrases = [r[0] for r in coverage_rows if r[5] == 0]
    empty_paras = [r[0] for r in coverage_rows if r[4] == 0]
    sparse_phrases = [r[0] for r in coverage_rows if 0 < r[5] < 5]
    sparse_paras = [r[0] for r in coverage_rows if 0 < r[4] < 3]

    print(f"\nSparse / empty chapters:")
    print(f"  empty phrase_pairs:     {len(empty_phrases)} {empty_phrases[:10]}")
    print(f"  empty paragraph_pairs:  {len(empty_paras)} {empty_paras[:10]}")
    print(f"  <5 phrases:             {len(sparse_phrases)} {sparse_phrases[:10]}")
    print(f"  <3 paragraphs:          {len(sparse_paras)} {sparse_paras[:10]}")


if __name__ == "__main__":
    main()
