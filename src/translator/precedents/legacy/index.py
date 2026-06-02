"""Build the on-disk precedent RAG index from the parallel corpus.

For each translated chapter (``part_NNN.jp.txt`` + ``part_NNN.en.txt``):

1. Split JP and EN into paragraphs
   (:func:`translator.precedents.align.split_jp_paragraphs` /
   :func:`translator.precedents.align.split_en_paragraphs`).
2. Run length-only DP alignment to produce monotonic JP↔EN paragraph
   pairs (:func:`translator.precedents.align.align_paragraphs`). No
   EN embedding is computed at any point in the build.
3. Embed *only* the JP side of each aligned pair; that's the row we
   later search against at retrieval time (the new chapter's JP
   queries dot-product against this matrix).

Persistence layout (``knowledge-vault/precedent-index/``):

* ``pairs.jsonl`` — one record per indexed entry::

      {
        "part_id": "part_004",
        "shape": "1:1" | "1:2" | "2:1",
        "jp_text": "…",
        "en_text": "…",
        "jp_idx_start": 0, "jp_idx_end": 0,    # inclusive paragraph range
        "en_idx_start": 0, "en_idx_end": 0,    # inclusive paragraph range
        "length_score": 0.93                    # 0..1 alignment quality
      }

* ``embeddings.npy`` — float32 ``(N, dim)`` matrix; row N matches the
  N-th line of ``pairs.jsonl``. Holds the JP-side embedding only.
* ``meta.json`` — provenance: ``{"model", "dim", "shape_counts",
  "parts_indexed", "length_ratio", "length_stdev", "last_updated"}``.

Build is incremental: re-running ``build_index`` on the same corpus
skips parts already present in ``meta.json["parts_indexed"]`` unless
``rebuild=True``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from translator.config import MODELS, PATHS, require_openai_key
from translator.precedents.legacy.align import (
    DEFAULT_LENGTH_RATIO,
    DEFAULT_LENGTH_VAR,
    ParagraphAlignment,
    align_paragraphs,
    split_en_paragraphs,
    split_jp_paragraphs,
)
from translator.prep.corpus import Part, iter_parts

# OpenAI's batch embedding endpoint accepts up to ~2048 inputs per
# call. We use 1024 to leave token-count headroom and keep individual
# requests bounded enough to retry cheaply on transient failures.
_EMBED_BATCH_SIZE = 1024

# Cap on chapter-level concurrency. The embedding endpoint allows much
# higher than this (10K RPM on default tier), but 16 keeps memory and
# error-blast-radius bounded.
_DEFAULT_INDEX_WORKERS = 16


@dataclass
class IndexEntry:
    """One row of the persisted ``pairs.jsonl``.

    ``embedding`` is held in memory while building; it's serialized
    separately into ``embeddings.npy`` keyed by row order.
    """

    part_id: str
    shape: str  # "1:1" | "1:2" | "2:1" — the paragraph-fan shape
    jp_text: str
    en_text: str
    jp_idx_start: int
    jp_idx_end: int
    en_idx_start: int
    en_idx_end: int
    length_score: float
    embedding: np.ndarray | None = None

    def to_record(self) -> dict:
        return {
            "part_id": self.part_id,
            "shape": self.shape,
            "jp_text": self.jp_text,
            "en_text": self.en_text,
            "jp_idx_start": self.jp_idx_start,
            "jp_idx_end": self.jp_idx_end,
            "en_idx_start": self.en_idx_start,
            "en_idx_end": self.en_idx_end,
            "length_score": float(self.length_score),
        }


@dataclass
class IndexStats:
    """Summary of a build invocation."""

    parts_indexed: list[str] = field(default_factory=list)
    parts_skipped: list[str] = field(default_factory=list)
    shape_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    model: str = ""
    dim: int = 0


def _index_paths(root: Path | None = None) -> tuple[Path, Path, Path]:
    base = root or PATHS.precedent_index
    return (
        base / "pairs.jsonl",
        base / "embeddings.npy",
        base / "meta.json",
    )


def _embed_texts(
    texts: list[str],
    *,
    model: str,
    client=None,
) -> np.ndarray:
    """Embed a flat list of texts and return an ``(N, D)`` float32 matrix.

    Embedding rows are L2-normalized so all downstream similarity
    work is plain dot products. A shared ``client`` can be passed in
    to amortize HTTP-pool setup when this function is called from
    many threads concurrently.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=require_openai_key())

    rows: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        # OpenAI API rejects empty strings; replace with a single
        # space so the batch alignment is preserved.
        sanitized = [t if t.strip() else " " for t in batch]
        resp = client.embeddings.create(model=model, input=sanitized)
        rows.extend(item.embedding for item in resp.data)

    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _load_existing(
    pairs_path: Path,
    embeddings_path: Path,
    meta_path: Path,
) -> tuple[list[dict], np.ndarray | None, dict]:
    """Load any prior index state. Returns ``([], None, {})`` when absent."""
    if not pairs_path.exists() or not embeddings_path.exists() or not meta_path.exists():
        return [], None, {}
    pairs: list[dict] = []
    with pairs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return pairs, embeddings, meta


def _alignments_to_entries(
    part_id: str,
    alignments: list[ParagraphAlignment],
) -> list[tuple[IndexEntry, str]]:
    """Convert each useful alignment into an ``(entry, embed_text)`` pair.

    1:0 and 0:1 records are dropped: a JP paragraph the translator
    deleted (or an EN paragraph the translator added) doesn't carry
    a usable precedent. 1:1 / 1:2 / 2:1 records are kept.

    The ``embed_text`` is the JP side concatenated with newlines —
    we embed the same text we'll later search against, so a 2:1
    record's stored vector represents the *combined* JP shape.
    """
    out: list[tuple[IndexEntry, str]] = []
    for align in alignments:
        a, b = align.shape
        if a == 0 or b == 0:
            continue
        shape_str = f"{a}:{b}"
        jp_text = "\n".join(p.text for p in align.jp_paragraphs)
        en_text = "\n".join(p.text for p in align.en_paragraphs)
        jp_idx_start = align.jp_paragraphs[0].index
        jp_idx_end = align.jp_paragraphs[-1].index
        en_idx_start = align.en_paragraphs[0].index
        en_idx_end = align.en_paragraphs[-1].index
        entry = IndexEntry(
            part_id=part_id,
            shape=shape_str,
            jp_text=jp_text,
            en_text=en_text,
            jp_idx_start=jp_idx_start,
            jp_idx_end=jp_idx_end,
            en_idx_start=en_idx_start,
            en_idx_end=en_idx_end,
            length_score=align.length_score,
        )
        out.append((entry, jp_text))
    return out


def _build_chapter_entries(
    part: Part,
    *,
    model: str,
    length_ratio: float,
    length_var: float,
    client=None,
) -> list[IndexEntry]:
    """Produce paragraph-level entries for a single translated chapter.

    No EN embedding is performed: alignment uses character-length DP
    only. Then we embed just the JP side of each kept alignment.
    """
    if part.en_text is None:
        return []
    jp_paragraphs = split_jp_paragraphs(part.jp_text)
    en_paragraphs = split_en_paragraphs(part.en_text)
    if not jp_paragraphs or not en_paragraphs:
        return []

    alignments = align_paragraphs(
        jp_paragraphs,
        en_paragraphs,
        length_ratio=length_ratio,
        length_var=length_var,
    )
    pairs = _alignments_to_entries(part.id, alignments)
    if not pairs:
        return []

    jp_texts_to_embed = [text for _, text in pairs]
    jp_emb = _embed_texts(jp_texts_to_embed, model=model, client=client)
    if jp_emb.size == 0:
        return []

    entries: list[IndexEntry] = []
    for (entry, _), embedding in zip(pairs, jp_emb, strict=True):
        entry.embedding = embedding
        entries.append(entry)
    return entries


def build_index(
    *,
    parts: Iterable[str] | None = None,
    rebuild: bool = False,
    model: str | None = None,
    length_ratio: float = DEFAULT_LENGTH_RATIO,
    length_var: float = DEFAULT_LENGTH_VAR,
    progress: bool = True,
    root: Path | None = None,
    workers: int = _DEFAULT_INDEX_WORKERS,
) -> IndexStats:
    """Build (or extend) the paragraph-level precedent index.

    Parameters
    ----------
    parts
        Iterable of part ids to include (e.g. ``("part_004", "part_017")``).
        ``None`` indexes every translated, supported-POV part.
    rebuild
        If True, discard any existing index on disk before building.
    model
        Override ``MODELS.embedding`` (used only for the JP side).
    length_ratio / length_var
        Calibration constants for the length-DP cost function
        (mean EN/JP char ratio and per-character residual variance).
        Defaults are the corpus-measured values. Override only for
        experimentation.
    progress
        Emit a one-line status per indexed chapter.
    root
        Override the on-disk index root. Useful for tests.
    workers
        Concurrent chapter workers (default 16).
    """
    out_root = root or PATHS.precedent_index
    out_root.mkdir(parents=True, exist_ok=True)
    pairs_path, emb_path, meta_path = _index_paths(out_root)
    chosen_model = model or MODELS.embedding

    if rebuild:
        for p in (pairs_path, emb_path, meta_path):
            if p.exists():
                p.unlink()

    existing_pairs, existing_emb, existing_meta = _load_existing(
        pairs_path, emb_path, meta_path
    )
    indexed_already: set[str] = set(existing_meta.get("parts_indexed", []))

    targets: list[Part] = []
    target_ids: set[str] | None = set(parts) if parts is not None else None
    for part in iter_parts(only_translated=True):
        if target_ids is not None and part.id not in target_ids:
            continue
        if part.id in indexed_already and not rebuild:
            continue
        targets.append(part)

    if not targets:
        # Nothing to do — but still report current state.
        shape_counts: dict[str, int] = {}
        for p in existing_pairs:
            shape_counts[p["shape"]] = shape_counts.get(p["shape"], 0) + 1
        return IndexStats(
            parts_indexed=sorted(indexed_already),
            parts_skipped=[],
            shape_counts=shape_counts,
            total_count=len(existing_pairs),
            model=str(existing_meta.get("model", chosen_model)),
            dim=int(
                existing_meta.get(
                    "dim",
                    existing_emb.shape[1] if existing_emb is not None and existing_emb.size > 0 else 0,
                )
            ),
        )

    from openai import OpenAI

    api_key = require_openai_key()

    def _build_one(part: Part) -> tuple[Part, list[IndexEntry]]:
        client = OpenAI(api_key=api_key)
        entries = _build_chapter_entries(
            part,
            model=chosen_model,
            length_ratio=length_ratio,
            length_var=length_var,
            client=client,
        )
        return part, entries

    by_part: dict[str, list[IndexEntry]] = {}
    parts_skipped: list[str] = []
    completed = 0
    n_targets = len(targets)
    effective_workers = max(1, min(workers, n_targets))
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_build_one, part): part for part in targets}
        for fut in as_completed(futures):
            part, entries = fut.result()
            completed += 1
            if not entries:
                parts_skipped.append(part.id)
                if progress:
                    print(
                        f"[precedents] [{completed}/{n_targets}] {part.id}: "
                        "skipped (no alignable paragraphs)"
                    )
                continue
            by_part[part.id] = entries
            if progress:
                shapes_here: dict[str, int] = {}
                for e in entries:
                    shapes_here[e.shape] = shapes_here.get(e.shape, 0) + 1
                shape_summary = ", ".join(
                    f"{count} {shape}" for shape, count in sorted(shapes_here.items())
                )
                print(
                    f"[precedents] [{completed}/{n_targets}] {part.id}: "
                    f"{len(entries)} entries ({shape_summary})"
                )

    # Reassemble new entries in canonical part order so re-builds are
    # deterministic and the on-disk pairs.jsonl stays human-readable.
    new_pairs: list[dict] = list(existing_pairs)
    new_embeddings: list[np.ndarray] = []
    if existing_emb is not None and existing_emb.size > 0:
        new_embeddings.append(existing_emb)
    parts_indexed_run: list[str] = []
    for part_id in sorted(by_part):
        entries = by_part[part_id]
        new_pairs.extend(e.to_record() for e in entries)
        new_embeddings.append(
            np.stack([e.embedding for e in entries if e.embedding is not None])
        )
        parts_indexed_run.append(part_id)

    if not parts_indexed_run:
        shape_counts = {}
        for p in new_pairs:
            shape_counts[p["shape"]] = shape_counts.get(p["shape"], 0) + 1
        return IndexStats(
            parts_indexed=sorted(indexed_already),
            parts_skipped=parts_skipped,
            shape_counts=shape_counts,
            total_count=len(new_pairs),
            model=chosen_model,
            dim=0,
        )

    full_embeddings = (
        np.concatenate(new_embeddings, axis=0)
        if new_embeddings
        else np.zeros((0, 0), dtype=np.float32)
    )
    pairs_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_pairs) + "\n",
        encoding="utf-8",
    )
    np.save(emb_path, full_embeddings)
    shape_counts = {}
    for p in new_pairs:
        shape_counts[p["shape"]] = shape_counts.get(p["shape"], 0) + 1
    parts_indexed_total = sorted(indexed_already.union(parts_indexed_run))
    meta = {
        "model": chosen_model,
        "dim": int(full_embeddings.shape[1]) if full_embeddings.size > 0 else 0,
        "shape_counts": shape_counts,
        "parts_indexed": parts_indexed_total,
        "length_ratio": length_ratio,
        "length_var": length_var,
        "last_updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return IndexStats(
        parts_indexed=parts_indexed_total,
        parts_skipped=parts_skipped,
        shape_counts=shape_counts,
        total_count=len(new_pairs),
        model=chosen_model,
        dim=meta["dim"],
    )


def load_meta(root: Path | None = None) -> dict:
    """Read the on-disk ``meta.json``; ``{}`` if not built yet."""
    _, _, meta_path = _index_paths(root)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def index_exists(root: Path | None = None) -> bool:
    """True iff a non-empty index has been built at ``root``."""
    pairs_path, emb_path, meta_path = _index_paths(root)
    if not (pairs_path.exists() and emb_path.exists() and meta_path.exists()):
        return False
    try:
        emb = np.load(emb_path, mmap_mode="r")
    except Exception:
        return False
    return emb.shape[0] > 0


__all__ = [
    "IndexEntry",
    "IndexStats",
    "build_index",
    "index_exists",
    "load_meta",
]
