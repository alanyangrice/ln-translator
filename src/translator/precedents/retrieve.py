"""Query-time retrieval against the precedent index.

For a target chapter (``part_NNN``):

1. Load its JP source.
2. Split into paragraphs (mirror of indexer's ``split_jp_paragraphs``).
3. Embed every JP paragraph query with the same model the index was
   built with.
4. Compute cosine similarity (L2-normalized rows → dot product) against
   the on-disk paragraph-level embedding matrix, filtered by holdout
   /self exclusion.
5. Return the top-K matches per query, dedupe across queries, cap by
   the per-chapter quota.

When the v2 index (``knowledge-vault/precedent-index-v2``) exists,
retrieval also runs a phrase-level pass: callers pass in a list of
:class:`~translator.precedents.risk.RiskEntry` objects from the
at-risk scanner, each flagged JP span is embedded, and the top-K
matching phrase precedents from the v2 phrase index are returned
alongside paragraph precedents.

Holdout / self exclusion is enforced by ``part_id`` filter so the
target chapter never retrieves its own precedents.

The retrieval call is intentionally cheap — embedding a chapter's
~50–100 paragraph queries costs <$0.002 — so it runs on every
translation, revision, and critique pass without budget concerns.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from translator.config import MODELS, PATHS, THRESHOLDS, require_openai_key
from translator.precedents.legacy.align import split_jp_paragraphs

if TYPE_CHECKING:
    from translator.precedents.risk import RiskEntry


@dataclass
class Precedent:
    """One retrieved paragraph-level precedent ready for prompt rendering."""

    part_id: str
    shape: str  # "1:1" | "1:2" | "2:1"
    jp_text: str
    en_text: str
    length_score: float
    # Score from the *current query* (cosine similarity between the
    # new chapter's JP paragraph and this stored precedent's JP
    # paragraph). Distinct from ``length_score`` which records how
    # confident we are in the original JP↔EN length-DP pairing.
    query_score: float = 0.0


@dataclass
class PhrasePrecedent:
    """One retrieved phrase-level precedent (v2 index only).

    Unlike paragraph precedents, phrase precedents are surgical: they
    cite a specific JP idiom and the human translator's rendering for
    that idiom alone (not surrounding context). The ``query_jp`` field
    records which at-risk span triggered this hit, so the prompt
    renderer can group the precedents by the original risk.

    ``query_literal_trap`` and ``query_category`` are copied from the
    triggering :class:`RiskEntry` so the prompt renderer can show *the
    risk's* trap (what the new chapter would translate to literally)
    rather than the matched precedent's literal_alternative (which is
    for a different surface form).
    """

    part_id: str
    jp_span: str
    en_span: str
    category: str
    literal_alternative: str
    notes: str
    context_jp: str
    query_jp: str = ""
    query_literal_trap: str = ""
    query_category: str = ""
    query_score: float = 0.0


@dataclass
class RetrievalResult:
    """Bundle of paragraph- and phrase-level precedents for a chapter."""

    part_id: str
    paragraphs: list[Precedent] = field(default_factory=list)
    phrases: list[PhrasePrecedent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Index version that produced this result, for prompt provenance
    # and downstream debugging.
    index_version: str = "v1"

    @property
    def is_empty(self) -> bool:
        return not self.paragraphs and not self.phrases

    @property
    def total(self) -> int:
        return len(self.paragraphs) + len(self.phrases)


@dataclass
class _LoadedIndex:
    """In-memory view of one on-disk index (single granularity)."""

    pairs: list[dict]
    embeddings: np.ndarray
    meta: dict


def _index_paths(root: Path | None = None) -> tuple[Path, Path, Path]:
    base = root or PATHS.precedent_index
    return base / "pairs.jsonl", base / "embeddings.npy", base / "meta.json"


def load_index(root: Path | None = None) -> _LoadedIndex | None:
    """Load the v1 on-disk index. Returns None if no index exists yet."""
    pairs_path, emb_path, meta_path = _index_paths(root)
    if not (pairs_path.exists() and emb_path.exists() and meta_path.exists()):
        return None
    pairs: list[dict] = []
    with pairs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    embeddings = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return _LoadedIndex(pairs=pairs, embeddings=embeddings, meta=meta)


def _v2_granularity_paths(
    root: Path, granularity: str
) -> tuple[Path, Path, Path]:
    base = root / granularity
    return base / "pairs.jsonl", base / "embeddings.npy", base / "meta.json"


def _load_v2_granularity(
    root: Path, granularity: str
) -> _LoadedIndex | None:
    """Load one granularity (``"phrases"`` or ``"paragraphs"``) of the v2 index."""
    pairs_path, emb_path, meta_path = _v2_granularity_paths(root, granularity)
    if not (pairs_path.exists() and emb_path.exists() and meta_path.exists()):
        return None
    pairs: list[dict] = []
    with pairs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return _LoadedIndex(
        pairs=pairs,
        embeddings=np.load(emb_path),
        meta=json.loads(meta_path.read_text(encoding="utf-8")),
    )


def v2_index_exists(root: Path | None = None) -> bool:
    """Whether the v2 multi-granular index is built and loadable."""
    base = root or PATHS.precedent_index_v2
    return (
        (base / "phrases" / "pairs.jsonl").exists()
        and (base / "paragraphs" / "pairs.jsonl").exists()
    )


def _embed_queries(texts: list[str], model: str) -> np.ndarray:
    """Embed query texts with L2-normalized rows. Same model as the index."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())
    # Single-call batch is fine: a chapter rarely exceeds ~150 paragraphs.
    resp = client.embeddings.create(
        model=model,
        input=[t if t.strip() else " " for t in texts],
    )
    arr = np.asarray([item.embedding for item in resp.data], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _topk_unique(
    query_emb: np.ndarray,
    target_emb: np.ndarray,
    pairs: list[dict],
    *,
    target_part_filter: np.ndarray,
    per_query_k: int,
    cap: int,
) -> list[tuple[int, float]]:
    """Return ``[(target_row, score), ...]`` deduped + capped at ``cap``.

    Each query selects its top-``per_query_k`` neighbors after
    filtering excluded ``part_id`` rows. Across queries we dedupe in
    two stages:

    1. By target row id (a single row never appears twice).
    2. By ``(jp_text, en_text)`` content — the same JP→EN pair
       stored from multiple chapters surfaces only once (the
       translator using the same phrase in two chapters is one
       precedent, not two). Different EN renderings of the same JP
       are kept (showing translation diversity is useful signal for
       the LLM).
    """
    if query_emb.size == 0 or target_emb.size == 0:
        return []
    if not target_part_filter.any():
        return []
    elig_idx = np.flatnonzero(target_part_filter)
    sub = target_emb[elig_idx]  # (T', D)
    sims = query_emb @ sub.T  # (Q, T')

    candidates: list[tuple[int, float]] = []
    seen_rows: set[int] = set()
    seen_text: set[tuple[str, str]] = set()
    for q in range(sims.shape[0]):
        row = sims[q]
        k = min(per_query_k, row.shape[0])
        if k <= 0:
            continue
        if k < row.shape[0]:
            part_idx = np.argpartition(-row, kth=k - 1)[:k]
        else:
            part_idx = np.arange(row.shape[0])
        # Sort the candidate slice by descending score.
        part_idx = part_idx[np.argsort(-row[part_idx])]
        for local_idx in part_idx:
            absolute = int(elig_idx[local_idx])
            if absolute in seen_rows:
                continue
            seen_rows.add(absolute)
            rec = pairs[absolute]
            text_key = (rec["jp_text"], rec["en_text"])
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            candidates.append((absolute, float(row[local_idx])))

    candidates.sort(key=lambda t: -t[1])
    if cap > 0:
        candidates = candidates[:cap]
    return candidates


def _retrieve_phrase_precedents_v2(
    *,
    risks: list["RiskEntry"],
    phrase_index: _LoadedIndex,
    excluded_parts: set[str],
    model: str,
    per_risk_k: int,
    total_cap: int,
    min_score: float,
) -> tuple[list[PhrasePrecedent], list[str]]:
    """Embed each at-risk JP span (with surrounding context for grounding)
    and retrieve the top-K matching phrase precedents from the v2 index.
    """
    notes: list[str] = []
    if not risks:
        return [], notes

    # Embed *just* the at-risk jp_span — the index was built from
    # jp_span-only vectors, and concatenating context_jp was empirically
    # shown to dilute the canonical-idiom signal (the top hit shifts
    # from the exact match to a paragraph-level cosine match that
    # happens to contain the same surface form in a different idiom).
    # See `tests/precedents/test_retrieval_smoke.py` (or the design
    # doc) for the head-to-head experiment.
    query_texts = [r.jp_span for r in risks]
    query_emb = _embed_queries(query_texts, model=model)
    if query_emb.size == 0:
        return [], notes

    # Build eligibility mask (exclude self/holdout chapters).
    elig_mask = np.array(
        [p["part_id"] not in excluded_parts for p in phrase_index.pairs],
        dtype=bool,
    )
    if not elig_mask.any():
        notes.append("phrase index empty after holdout filter")
        return [], notes

    elig_idx = np.flatnonzero(elig_mask)
    sub_emb = phrase_index.embeddings[elig_idx]
    sims = query_emb @ sub_emb.T  # (Q, T')

    seen_keys: set[tuple[str, str]] = set()
    candidates: list[PhrasePrecedent] = []
    for q, risk in enumerate(risks):
        row = sims[q]
        k = min(per_risk_k, row.shape[0])
        if k <= 0:
            continue
        if k < row.shape[0]:
            top_local = np.argpartition(-row, kth=k - 1)[:k]
        else:
            top_local = np.arange(row.shape[0])
        top_local = top_local[np.argsort(-row[top_local])]
        for local_idx in top_local:
            score = float(row[local_idx])
            if score < min_score:
                break
            absolute = int(elig_idx[local_idx])
            rec = phrase_index.pairs[absolute]
            key = (rec.get("jp_span", ""), rec.get("en_span", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                PhrasePrecedent(
                    part_id=rec["part_id"],
                    jp_span=rec.get("jp_span", ""),
                    en_span=rec.get("en_span", ""),
                    category=rec.get("category", "other"),
                    literal_alternative=rec.get("literal_alternative", ""),
                    notes=rec.get("notes", ""),
                    context_jp=rec.get("context_jp", ""),
                    query_jp=risk.jp_span,
                    query_literal_trap=risk.literal_trap,
                    query_category=risk.category,
                    query_score=score,
                )
            )

    # Cap the global volume across all risks. Keep highest-scoring.
    if total_cap > 0 and len(candidates) > total_cap:
        candidates.sort(key=lambda p: -p.query_score)
        dropped = len(candidates) - total_cap
        candidates = candidates[:total_cap]
        notes.append(
            f"capped phrase precedents at {total_cap} "
            f"(dropped {dropped} lowest-scoring matches)"
        )

    return candidates, notes


def _retrieve_paragraphs_v2(
    *,
    part_id: str,
    paragraph_index: _LoadedIndex,
    excluded_parts: set[str],
    model: str,
    per_query_k: int,
    cap: int,
) -> tuple[list[Precedent], list[str]]:
    """Paragraph retrieval against the v2 index (no length/semantic filters
    since v2 entries are LLM-extracted with quality gates instead).
    """
    from translator.prep.corpus import load_part_jp

    notes: list[str] = []
    jp_text = load_part_jp(part_id)
    jp_paragraphs = split_jp_paragraphs(jp_text)
    if not jp_paragraphs:
        notes.append("target chapter produced no JP paragraphs after splitting")
        return [], notes

    query_emb = _embed_queries(
        [p.text for p in jp_paragraphs], model=model
    )

    elig_mask = np.array(
        [p["part_id"] not in excluded_parts for p in paragraph_index.pairs],
        dtype=bool,
    )
    if not elig_mask.any():
        notes.append("paragraph index empty after holdout filter")
        return [], notes

    hits = _topk_unique(
        query_emb,
        paragraph_index.embeddings,
        paragraph_index.pairs,
        target_part_filter=elig_mask,
        per_query_k=per_query_k,
        cap=cap,
    )
    out: list[Precedent] = []
    for row, score in hits:
        rec = paragraph_index.pairs[row]
        out.append(
            Precedent(
                part_id=rec["part_id"],
                shape=rec.get("shape", "1:1"),
                jp_text=rec.get("jp_text", ""),
                en_text=rec.get("en_text", ""),
                # v2 entries have no length_score; report 1.0 so
                # downstream rendering knows they passed the LLM
                # quality gates.
                length_score=1.0,
                query_score=score,
            )
        )
    return out, notes


def retrieve_for_part(
    part_id: str,
    *,
    paragraph_k: int | None = None,
    per_query_k: int | None = None,
    min_length_score: float | None = None,
    min_semantic_score: float | None = None,
    exclude: Iterable[str] | None = None,
    model: str | None = None,
    root: Path | None = None,
    risks: list["RiskEntry"] | None = None,
    phrase_per_risk_k: int | None = None,
    phrase_total_cap: int | None = None,
    phrase_min_score: float | None = None,
) -> RetrievalResult:
    """Retrieve precedents for ``part_id``.

    Routes to the v2 multi-granular index (phrase + paragraph) when
    available, falls back to v1 (length-DP paragraph only) otherwise.

    Parameters
    ----------
    part_id
        Target chapter id (e.g. ``"part_017"``).
    paragraph_k
        Per-chapter paragraph cap (default
        :class:`Thresholds.precedents_per_chapter`).
    per_query_k
        How many paragraph neighbors each JP paragraph asks the index
        for (default :class:`Thresholds.precedents_top_k_per_query`).
    min_length_score, min_semantic_score
        v1-only filters. Ignored when retrieving against v2 since v2
        entries are LLM-extracted with quality gates instead of
        length-DP scores.
    exclude
        Additional part ids to exclude from retrieval. The target
        ``part_id`` itself is **always** excluded so a chapter never
        retrieves its own precedents.
    risks
        v2-only. List of :class:`RiskEntry` from the at-risk scanner.
        When provided, each risk's ``jp_span`` is used as a phrase-level
        query against the v2 phrase index; results populate
        ``RetrievalResult.phrases``. When None, no phrase retrieval runs.
    phrase_per_risk_k, phrase_total_cap, phrase_min_score
        v2-only knobs (default
        :class:`Thresholds.phrase_precedents_per_risk` /
        ``phrase_precedents_total_cap`` / ``phrase_precedents_min_score``).
    """
    cap = paragraph_k if paragraph_k is not None else THRESHOLDS.precedents_per_chapter
    pq_k = per_query_k if per_query_k is not None else THRESHOLDS.precedents_top_k_per_query
    chosen_model = model or MODELS.embedding

    excluded: set[str] = set(exclude or ())
    excluded.add(part_id)

    # ── v2 path ─────────────────────────────────────────────────
    v2_root = root or PATHS.precedent_index_v2
    if v2_index_exists(v2_root):
        return _retrieve_for_part_v2(
            part_id=part_id,
            v2_root=v2_root,
            excluded=excluded,
            paragraph_k=cap,
            per_query_k=pq_k,
            phrase_per_risk_k=phrase_per_risk_k or THRESHOLDS.phrase_precedents_per_risk,
            phrase_total_cap=phrase_total_cap or THRESHOLDS.phrase_precedents_total_cap,
            phrase_min_score=(
                phrase_min_score
                if phrase_min_score is not None
                else THRESHOLDS.phrase_precedents_min_score
            ),
            risks=risks,
            model=chosen_model,
        )

    # ── v1 fallback ─────────────────────────────────────────────
    from translator.prep.corpus import load_part_jp

    score_floor = (
        min_length_score
        if min_length_score is not None
        else THRESHOLDS.precedents_min_length_score
    )
    semantic_floor = (
        min_semantic_score
        if min_semantic_score is not None
        else THRESHOLDS.precedents_min_semantic_score
    )

    notes: list[str] = []
    result = RetrievalResult(part_id=part_id, index_version="v1")

    index = load_index(root=root)
    if index is None:
        notes.append(
            "no precedent index built — run `translator precedents build`"
        )
        result.notes = notes
        return result
    if index.embeddings.size == 0:
        notes.append("precedent index is empty")
        result.notes = notes
        return result

    if index.meta.get("model") and index.meta["model"] != chosen_model:
        notes.append(
            f"index built with {index.meta['model']!r} but query model is "
            f"{chosen_model!r}; pin MODEL_EMBEDDING in .env or rebuild"
        )

    def _eligible(p: dict) -> bool:
        if p["part_id"] in excluded:
            return False
        if float(p.get("length_score", 0.0)) < score_floor:
            return False
        # ``semantic_score`` only present if validate has run; pairs
        # without a score bypass this filter so retrieval stays usable
        # before validation completes.
        if "semantic_score" in p and float(p["semantic_score"]) < semantic_floor:
            return False
        return True

    target_part_filter = np.array(
        [_eligible(p) for p in index.pairs], dtype=bool
    )
    n_length_drop = sum(
        1
        for p in index.pairs
        if p["part_id"] not in excluded
        and float(p.get("length_score", 0.0)) < score_floor
    )
    n_semantic_drop = sum(
        1
        for p in index.pairs
        if p["part_id"] not in excluded
        and float(p.get("length_score", 0.0)) >= score_floor
        and "semantic_score" in p
        and float(p["semantic_score"]) < semantic_floor
    )
    if n_length_drop:
        notes.append(
            f"filtered {n_length_drop} candidate(s) below length-score "
            f"threshold {score_floor:.2f}"
        )
    if n_semantic_drop:
        notes.append(
            f"filtered {n_semantic_drop} additional candidate(s) below "
            f"semantic-score threshold {semantic_floor:.2f}"
        )
    n_unvalidated = sum(
        1 for p in index.pairs if "semantic_score" not in p
    )
    if n_unvalidated and semantic_floor > 0:
        notes.append(
            f"{n_unvalidated} pair(s) lack semantic_score (run "
            "`translator precedents validate` to score them)"
        )
    if not target_part_filter.any():
        notes.append("all index entries excluded after holdout/self filter")
        result.notes = notes
        return result

    # Build JP paragraph queries from the target chapter.
    jp_text = load_part_jp(part_id)
    jp_paragraphs = split_jp_paragraphs(jp_text)
    if not jp_paragraphs:
        notes.append("target chapter produced no JP paragraphs after splitting")
        result.notes = notes
        return result

    query_emb = _embed_queries(
        [p.text for p in jp_paragraphs], model=chosen_model
    )

    hits = _topk_unique(
        query_emb,
        index.embeddings,
        index.pairs,
        target_part_filter=target_part_filter,
        per_query_k=pq_k,
        cap=cap,
    )
    for row, score in hits:
        rec = index.pairs[row]
        result.paragraphs.append(
            Precedent(
                part_id=rec["part_id"],
                shape=rec.get("shape", "1:1"),
                jp_text=rec["jp_text"],
                en_text=rec["en_text"],
                length_score=float(rec.get("length_score", 0.0)),
                query_score=score,
            )
        )

    if not result.paragraphs:
        notes.append("no precedents matched (index may be too small or model mismatched)")

    result.notes = notes
    return result


def _retrieve_for_part_v2(
    *,
    part_id: str,
    v2_root: Path,
    excluded: set[str],
    paragraph_k: int,
    per_query_k: int,
    phrase_per_risk_k: int,
    phrase_total_cap: int,
    phrase_min_score: float,
    risks: list["RiskEntry"] | None,
    model: str,
) -> RetrievalResult:
    """v2 multi-granular retrieval entry point. Phrase retrieval is
    risk-driven (only runs when ``risks`` is provided)."""
    notes: list[str] = []
    result = RetrievalResult(part_id=part_id, index_version="v2")

    paragraph_index = _load_v2_granularity(v2_root, "paragraphs")
    phrase_index = _load_v2_granularity(v2_root, "phrases")

    if paragraph_index is None and phrase_index is None:
        notes.append("v2 index directory exists but neither granularity loaded")
        result.notes = notes
        return result

    # Sanity-check embedding-model match (both granularities use the same).
    for idx, label in (
        (paragraph_index, "paragraph"),
        (phrase_index, "phrase"),
    ):
        if idx and idx.meta.get("embedding_model") and idx.meta["embedding_model"] != model:
            notes.append(
                f"v2 {label} index built with {idx.meta['embedding_model']!r} "
                f"but query model is {model!r}; pin MODEL_EMBEDDING or rebuild"
            )

    # ── paragraph pass ──────────────────────────────────────────
    if paragraph_index is not None:
        paragraphs, p_notes = _retrieve_paragraphs_v2(
            part_id=part_id,
            paragraph_index=paragraph_index,
            excluded_parts=excluded,
            model=model,
            per_query_k=per_query_k,
            cap=paragraph_k,
        )
        result.paragraphs = paragraphs
        notes.extend(p_notes)
    else:
        notes.append("v2 paragraph index missing")

    # ── phrase pass (only if risks were supplied) ──────────────
    if risks and phrase_index is not None:
        phrases, ph_notes = _retrieve_phrase_precedents_v2(
            risks=risks,
            phrase_index=phrase_index,
            excluded_parts=excluded,
            model=model,
            per_risk_k=phrase_per_risk_k,
            total_cap=phrase_total_cap,
            min_score=phrase_min_score,
        )
        result.phrases = phrases
        notes.extend(ph_notes)
    elif risks and phrase_index is None:
        notes.append("v2 phrase index missing")

    if not result.paragraphs and not result.phrases:
        notes.append("no v2 precedents matched")

    result.notes = notes
    return result


def index_exists() -> bool:  # noqa: D401 - thin convenience
    """Whether *some* precedent index (v1 or v2) is present and queryable."""
    return v2_index_exists() or load_index() is not None


__all__ = [
    "PhrasePrecedent",
    "Precedent",
    "RetrievalResult",
    "index_exists",
    "load_index",
    "retrieve_for_part",
    "v2_index_exists",
]
