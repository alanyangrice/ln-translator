"""Deep-audit the v2 precedent index for content-level shortcuts.

The shallow audit (``audit_index_v2.py``) only checks coverage and
representation. This script adds:

  A. **Sub-threshold mega-pair detection** — anything in [400, 800] JP
     chars or [1000, 2000] EN chars is "borderline lazy". We measure
     the per-chapter rate.
  B. **Hallucinated EN-span check** — does each phrase pair's
     ``en_span`` actually appear verbatim in the source EN translation?
  C. **Phrase trivia detection** — what fraction of phrase pairs have
     ``literal_alternative`` essentially equal to ``en_span``? These
     are vocabulary substitutions masquerading as idioms.
  D. **Order check** — for each chapter, do paragraph pairs appear in
     source order (i.e. each successive pair's first JP fragment
     appears later in the source than the previous)?
  E. **JP-span fidelity** — do phrase ``jp_span`` values appear
     verbatim in the source JP?
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterable

from translator.precedents.extract import default_v2_root
from translator.prep.corpus import load_part_jp, load_part_en, iter_parts


def _normalize_compact(s: str) -> str:
    """Whitespace-stripped lowercase for fuzzy in-text checks."""
    return "".join(s.split()).lower()


def main() -> None:
    root = default_v2_root()
    print(f"Index root: {root}\n")

    # Universe of parallel chapters.
    parallel = []
    for part in iter_parts(only_translated=True, parts_only=True):
        if load_part_en(part.id) is not None:
            parallel.append(part.id)
    parallel_set = set(parallel)

    # Load index.
    phrases = [
        json.loads(l)
        for l in (root / "phrases" / "pairs.jsonl")
            .read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    paragraphs = [
        json.loads(l)
        for l in (root / "paragraphs" / "pairs.jsonl")
            .read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    phrases_by_part: dict[str, list[dict]] = defaultdict(list)
    paras_by_part: dict[str, list[dict]] = defaultdict(list)
    for p in phrases:
        phrases_by_part[p["part_id"]].append(p)
    for p in paragraphs:
        paras_by_part[p["part_id"]].append(p)

    # ── A. Sub-threshold mega-pair detection ──────────────────────
    print("A) Sub-threshold borderline-lazy paragraph pairs")
    print("   (per-chapter share of pairs with jp ∈ [400,800] or en ∈ [1000,2000])")
    chap_borderline_rate: list[tuple[str, float, int]] = []
    for pid in parallel:
        pairs = paras_by_part.get(pid, [])
        if not pairs:
            continue
        borderline = sum(
            1 for p in pairs
            if 400 <= len(p["jp_text"]) <= 800
            or 1000 <= len(p["en_text"]) <= 2000
        )
        chap_borderline_rate.append((pid, borderline / len(pairs), len(pairs)))

    rates = [r[1] for r in chap_borderline_rate]
    print(f"   mean={statistics.mean(rates):.2%}  median={statistics.median(rates):.2%}  "
          f"max={max(rates):.2%}")
    worst = sorted(chap_borderline_rate, key=lambda x: -x[1])[:10]
    print("   Top 10 chapters by borderline-pair rate:")
    for pid, rate, npairs in worst:
        flag = "  ⚠️" if rate > 0.30 else ""
        print(f"     {pid}: {rate:.0%}  ({int(rate * npairs)}/{npairs} pairs){flag}")

    # ── B. Hallucinated EN-span check ─────────────────────────────
    print("\nB) Phrase EN-span verbatim presence in source EN")
    miss_by_part: dict[str, int] = Counter()
    total_by_part: dict[str, int] = Counter()
    sample_misses: list[tuple[str, str, str]] = []
    en_cache: dict[str, str] = {}
    for ph in phrases:
        pid = ph["part_id"]
        if pid not in en_cache:
            en_text = load_part_en(pid) or ""
            en_cache[pid] = _normalize_compact(en_text)
        total_by_part[pid] += 1
        compact_span = _normalize_compact(ph["en_span"])
        if compact_span and compact_span not in en_cache[pid]:
            miss_by_part[pid] += 1
            if len(sample_misses) < 12:
                sample_misses.append((pid, ph["en_span"][:80], ph["jp_span"][:60]))

    total_phrases = sum(total_by_part.values())
    total_misses = sum(miss_by_part.values())
    print(f"   total phrase pairs: {total_phrases}")
    print(f"   en_span missing from source EN: {total_misses} "
          f"({total_misses / max(1, total_phrases):.1%})")
    if miss_by_part:
        worst_part = sorted(miss_by_part.items(), key=lambda x: -x[1])[:8]
        print("   Top 8 chapters by hallucination count:")
        for pid, n in worst_part:
            print(f"     {pid}: {n}/{total_by_part[pid]} ({n / total_by_part[pid]:.0%}) phrases")
    print("   Sample missed en_spans (jp → en, not present verbatim):")
    for pid, en_s, jp_s in sample_misses[:6]:
        print(f"     [{pid}] {jp_s} → {en_s}")

    # ── C. Phrase trivia detection ────────────────────────────────
    print("\nC) Phrase 'trivia' rate (literal_alt ≈ en_span = vocab substitution)")
    trivia_count = 0
    trivia_samples: list[tuple[str, str, str]] = []
    for ph in phrases:
        en = _normalize_compact(ph["en_span"])
        lit = _normalize_compact(ph["literal_alternative"])
        if not en or not lit:
            continue
        # Trivia heuristics: literal_alt is the same as en_span
        # (modulo trivial differences), or one is contained in the other.
        if en == lit or (len(en) > 0 and en in lit) or (len(lit) > 0 and lit in en):
            # but only flag if both are short (otherwise legit reordering)
            if max(len(en), len(lit)) <= 30:
                trivia_count += 1
                if len(trivia_samples) < 8:
                    trivia_samples.append(
                        (ph["part_id"], ph["en_span"], ph["literal_alternative"])
                    )
    print(f"   trivia phrases: {trivia_count} / {total_phrases} "
          f"({trivia_count / max(1, total_phrases):.1%})")
    print("   Samples (en_span vs literal_alternative):")
    for pid, en_s, lit_s in trivia_samples:
        print(f"     [{pid}] '{en_s}' vs '{lit_s}'")

    # ── D. Paragraph order check ──────────────────────────────────
    print("\nD) Paragraph order check (jp pairs should appear in source order)")
    out_of_order_chapters: list[tuple[str, int, int]] = []
    for pid, pairs in paras_by_part.items():
        if len(pairs) < 2:
            continue
        src = load_part_jp(pid)
        # Find each pair's first occurrence in source.
        positions = []
        for p in pairs:
            head = p["jp_text"].split("\n")[0][:30].strip()
            if head:
                idx = src.find(head)
                positions.append(idx)
            else:
                positions.append(-1)
        # Count inversions in valid positions.
        valid = [(i, pos) for i, pos in enumerate(positions) if pos >= 0]
        inv = sum(
            1 for i in range(len(valid) - 1)
            if valid[i][1] > valid[i + 1][1]
        )
        if inv > 0:
            out_of_order_chapters.append((pid, inv, len(valid)))
    out_of_order_chapters.sort(key=lambda x: -x[1])
    print(f"   chapters with order inversions: {len(out_of_order_chapters)}")
    for pid, inv, n in out_of_order_chapters[:10]:
        print(f"     {pid}: {inv} inversions across {n} pairs")

    # ── E. JP-span fidelity ───────────────────────────────────────
    print("\nE) Phrase JP-span verbatim presence in source JP")
    jp_misses = 0
    jp_miss_samples: list[tuple[str, str]] = []
    jp_cache: dict[str, str] = {}
    for ph in phrases:
        pid = ph["part_id"]
        if pid not in jp_cache:
            jp_cache[pid] = _normalize_compact(load_part_jp(pid))
        if _normalize_compact(ph["jp_span"]) not in jp_cache[pid]:
            jp_misses += 1
            if len(jp_miss_samples) < 8:
                jp_miss_samples.append((pid, ph["jp_span"]))
    print(f"   jp_span missing from source JP: {jp_misses} / {total_phrases} "
          f"({jp_misses / max(1, total_phrases):.2%})")
    for pid, jp_s in jp_miss_samples[:6]:
        print(f"     [{pid}] {jp_s}")


if __name__ == "__main__":
    main()
