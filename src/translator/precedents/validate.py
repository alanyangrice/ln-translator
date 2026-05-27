"""Cross-lingual validation pass over the precedent index.

Length-only DP alignment is fast and structurally correct
(monotonic, handles 1:N / N:1) but blind to content: when two
adjacent paragraphs have similar character lengths, the DP can lock
onto the wrong neighbor. The resulting JP↔EN pair has a high
``length_score`` but the EN side doesn't actually correspond to the
JP side.

This module fixes that by computing a *post-hoc* cross-lingual
cosine score for every stored pair. ``text-embedding-3-small`` is
multilingual, so for a real translation pair the JP and EN
embeddings cluster high (~0.5–0.8); for a misaligned pair they
collapse to near-zero. The resulting ``semantic_score`` is written
back into ``pairs.jsonl`` and used as a second quality filter at
retrieval time, stacked on top of ``length_score``::

    eligible = (length_score >= 0.3) AND (semantic_score >= 0.4)

Crucially: EN embeddings are computed once during validation and
**discarded immediately**. They never enter ``embeddings.npy`` and
never load at retrieval time — the hot path remains JP-only.

Calibration: chapters with exactly matching JP/EN paragraph counts
(14 of 229 in this corpus) are by-construction correctly aligned.
Their ``semantic_score`` distribution defines the "true positive"
band; the 5th percentile of that distribution is the recommended
filter threshold for the rest of the corpus.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from translator.config import MODELS, PATHS, require_openai_key
from translator.precedents.index import _embed_texts, _index_paths

# Larger batch is fine here: we're doing a one-shot pass and
# OpenAI's embedding endpoint accepts up to 2048 inputs per call.
# Each pair embeds at most one paragraph (or a 2-paragraph merge),
# so 1024 keeps token-count well under the per-request limit.
_VALIDATION_BATCH_SIZE = 1024
_DEFAULT_WORKERS = 8


@dataclass
class ValidationStats:
    """Summary of a validate run."""

    pairs_validated: int = 0
    pairs_skipped: int = 0
    semantic_score_mean: float = 0.0
    semantic_score_median: float = 0.0
    semantic_score_p05: float = 0.0
    semantic_score_p10: float = 0.0
    semantic_score_p25: float = 0.0
    suggested_threshold: float = 0.0
    # Calibration on chapters where alignment is correct by construction.
    calibration_pairs: int = 0
    calibration_mean: float = 0.0
    calibration_median: float = 0.0
    calibration_p05: float = 0.0
    # Pairs that pass length_score >= 0.3 but fail semantic threshold.
    suspect_count: int = 0
    suspect_examples: list[dict] = field(default_factory=list)


def _load_index_for_validation(
    pairs_path: Path,
    embeddings_path: Path,
    meta_path: Path,
) -> tuple[list[dict], np.ndarray, dict]:
    pairs: list[dict] = []
    with pairs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    embeddings = np.load(embeddings_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(pairs):
        raise RuntimeError(
            f"embedding/pair row count mismatch: {embeddings.shape[0]} vs {len(pairs)}"
        )
    return pairs, embeddings, meta


def _calibration_part_ids(corpus_root: Path | None = None) -> set[str]:
    """Return part ids whose JP/EN paragraph counts match exactly.

    These are alignment-correct by construction (no merge/split
    ambiguity), so their semantic-score distribution defines the
    "true positive" band. The list is recomputed from disk rather
    than cached so it stays accurate as new chapters are translated.
    """
    parallel_root = corpus_root or PATHS.parallel
    out: set[str] = set()
    for jp_path in sorted(parallel_root.glob("part_*.jp.txt")):
        en_path = jp_path.with_name(jp_path.name.replace(".jp.txt", ".en.txt"))
        if not en_path.exists():
            continue
        jp_count = sum(
            1 for line in jp_path.read_text().splitlines() if line.strip()
        )
        en_count = sum(
            1 for line in en_path.read_text().splitlines() if line.strip()
        )
        if jp_count == en_count and jp_count > 0:
            out.add(jp_path.stem.replace(".jp", ""))
    return out


def _embed_en_batch(
    texts: list[str], *, model: str, api_key: str
) -> np.ndarray:
    """Per-thread embedding wrapper. Each thread uses its own
    ``OpenAI`` client so connection pools don't share mutable state.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    return _embed_texts(texts, model=model, client=client)


def validate_index(
    *,
    rebuild: bool = False,
    workers: int = _DEFAULT_WORKERS,
    progress: bool = True,
    root: Path | None = None,
) -> ValidationStats:
    """Embed every pair's EN side and write ``semantic_score`` back.

    Parameters
    ----------
    rebuild
        If True, recompute ``semantic_score`` for all pairs even when
        already present. Default False = incremental (skip pairs that
        already have a score).
    workers
        Concurrent embedding workers. Each handles one batch of
        ``_VALIDATION_BATCH_SIZE`` EN texts. Default 8.
    progress
        Print a one-line status per batch.
    root
        Override the on-disk index root. Useful for tests.
    """
    out_root = root or PATHS.precedent_index
    pairs_path, emb_path, meta_path = _index_paths(out_root)
    if not (pairs_path.exists() and emb_path.exists() and meta_path.exists()):
        raise RuntimeError(
            f"no precedent index at {out_root}; run `translator precedents build` first"
        )

    pairs, jp_embeddings, meta = _load_index_for_validation(
        pairs_path, emb_path, meta_path
    )
    model = str(meta.get("model", MODELS.embedding))
    api_key = require_openai_key()

    # Identify pairs to (re)validate.
    todo_indices: list[int] = []
    for idx, rec in enumerate(pairs):
        if rebuild or "semantic_score" not in rec:
            todo_indices.append(idx)

    if not todo_indices:
        if progress:
            print(
                f"[validate] all {len(pairs)} pairs already have semantic_score; "
                "nothing to do (use --rebuild to recompute)"
            )
        return _summarize(pairs, todo_count=0)

    if progress:
        print(
            f"[validate] {len(todo_indices)} pair(s) to embed (model={model}, "
            f"workers={workers}, batch={_VALIDATION_BATCH_SIZE})"
        )

    # Build batches keyed by their absolute index range.
    batches: list[tuple[list[int], list[str]]] = []
    for start in range(0, len(todo_indices), _VALIDATION_BATCH_SIZE):
        chunk = todo_indices[start : start + _VALIDATION_BATCH_SIZE]
        batches.append((chunk, [pairs[i]["en_text"] for i in chunk]))

    completed = 0
    n_batches = len(batches)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, n_batches))) as pool:
        futures = {
            pool.submit(_embed_en_batch, texts, model=model, api_key=api_key): (
                indices,
                texts,
            )
            for indices, texts in batches
        }
        for fut in as_completed(futures):
            indices, _ = futures[fut]
            en_emb = fut.result()  # already L2-normalized
            # Cross-lingual cosine: stored JP rows are L2-normalized,
            # so dot product is the cosine.
            jp_rows = jp_embeddings[np.asarray(indices)]
            cosines = np.einsum("ij,ij->i", jp_rows, en_emb)
            for idx, score in zip(indices, cosines, strict=True):
                pairs[idx]["semantic_score"] = float(score)
            completed += 1
            if progress:
                print(
                    f"[validate] [{completed}/{n_batches}] "
                    f"{len(indices)} pairs scored "
                    f"(this batch median={float(np.median(cosines)):.3f})"
                )

    # Persist back to disk. Re-write pairs.jsonl in row order (matches
    # embeddings.npy by construction; the line index is the
    # embedding row index).
    pairs_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in pairs) + "\n",
        encoding="utf-8",
    )

    # Update meta with provenance for the validation pass.
    meta["validated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return _summarize(pairs, todo_count=len(todo_indices))


def _summarize(pairs: list[dict], *, todo_count: int) -> ValidationStats:
    """Compute calibration + corpus-wide stats from the in-memory pair list."""
    scores: list[float] = [
        float(p["semantic_score"])
        for p in pairs
        if "semantic_score" in p
    ]
    if not scores:
        return ValidationStats(pairs_validated=todo_count)

    sorted_scores = sorted(scores)

    def percentile(p: float) -> float:
        if not sorted_scores:
            return 0.0
        idx = max(0, min(len(sorted_scores) - 1, int(len(sorted_scores) * p)))
        return sorted_scores[idx]

    # Calibration: only the chapters with exact paragraph-count match.
    calibration_parts = _calibration_part_ids()
    calibration_scores = [
        float(p["semantic_score"])
        for p in pairs
        if p["part_id"] in calibration_parts
        and "semantic_score" in p
        and p["shape"] == "1:1"
    ]
    if calibration_scores:
        calibration_sorted = sorted(calibration_scores)
        cal_p05 = calibration_sorted[max(0, len(calibration_sorted) // 20)]
    else:
        cal_p05 = 0.0
        calibration_sorted = []

    # Suspect band: pairs that survived length-score floor (0.3) but
    # would fail the suggested semantic threshold. These are the
    # length-DP false positives — adjacent paragraphs that happen to
    # have similar character counts.
    suspect = [
        p
        for p in pairs
        if float(p.get("length_score", 0.0)) >= 0.3
        and float(p.get("semantic_score", 1.0)) < cal_p05
    ]

    return ValidationStats(
        pairs_validated=todo_count,
        pairs_skipped=len(pairs) - len(scores),
        semantic_score_mean=statistics.mean(scores),
        semantic_score_median=statistics.median(scores),
        semantic_score_p05=percentile(0.05),
        semantic_score_p10=percentile(0.10),
        semantic_score_p25=percentile(0.25),
        suggested_threshold=round(cal_p05, 3),
        calibration_pairs=len(calibration_scores),
        calibration_mean=statistics.mean(calibration_scores) if calibration_scores else 0.0,
        calibration_median=statistics.median(calibration_scores) if calibration_scores else 0.0,
        calibration_p05=cal_p05,
        suspect_count=len(suspect),
        suspect_examples=[
            {
                "part_id": p["part_id"],
                "shape": p["shape"],
                "length_score": p["length_score"],
                "semantic_score": p["semantic_score"],
                "jp_text": p["jp_text"][:120],
                "en_text": p["en_text"][:120],
            }
            for p in sorted(suspect, key=lambda x: x["semantic_score"])[:5]
        ],
    )


__all__ = ["ValidationStats", "validate_index"]
