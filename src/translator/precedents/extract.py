"""LLM-extracted multi-granular precedent extraction.

Replaces the length-only DP alignment in :mod:`translator.precedents.align`
with structured extraction by an external LLM (DeepSeek V4-Flash by
default, 1M context / 384K output / $0.14-$0.28 per 1M tokens; any
model the provider abstraction can route also works). For each chapter
pair ``(part_NNN.jp.txt, part_NNN.en.txt)`` we ask the model to emit
two kinds of aligned precedents:

1. **phrase_pairs** — JP idioms / set expressions / restructured grammar
   alongside the established translator's idiomatic EN rendering. These
   are the surgical instruments: when a new chapter contains the same JP
   phrase, we can inject a high-precision hint pointing the translator
   at the natural EN form (and away from the literal). The killer feature
   that paragraph-level cosine retrieval consistently misses because the
   idiom signal gets diluted by surrounding context.

2. **paragraph_pairs** — whole-paragraph alignments for voice/rhythm
   matching. Conceptually the same shape as the v1 length-DP entries
   but produced by an LLM that handles 1:N, N:1, and N:M mappings
   without a length prior, sidestepping the false-positive issues
   length-DP exhibited (e.g. an aligned pair scoring length=0.43 /
   semantic=0.38 because the EN naturally condenses).

Persistence layout (default ``knowledge-vault/precedent-index-v2/``)::

    cache/
        {part_id}.json        # raw LLM output, idempotent reuse
    phrases/
        pairs.jsonl
        embeddings.npy
        meta.json
    paragraphs/
        pairs.jsonl
        embeddings.npy
        meta.json
    meta.json                 # global build provenance

The cache layer is intentional: the LLM extraction is the expensive
step (DeepSeek tokens). Re-running ``extract_corpus`` on the same
chapters is free if the cache hits, which makes incremental rebuilds
(after fixing the prompt, swapping models, etc.) cheap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from translator.config import MODELS, VAULT_DIR, require_openai_key
from translator.inference.provider import complete
from translator.precedents.prompts import load_template
from translator.prep.corpus import iter_parts, load_part_en, load_part_jp

# OpenAI batch embedding ceiling (same constant as v1).
_EMBED_BATCH_SIZE = 1024

# Default extraction model. DeepSeek V4-Flash is ~10x cheaper than
# Sonnet 4.6 ($0.14/M input, $0.28/M output as of May 2026) and
# supports up to 384K output tokens with a 1M context window —
# plenty of headroom for full-chapter extraction, including
# voluminous paragraph alignments. Override with ``--model
# deepseek-v4-pro`` when the chapter is exceptionally tricky;
# Pro is ~3x the cost of Flash but still well below Sonnet.
DEFAULT_EXTRACTION_MODEL = "deepseek-v4-flash"

# Cap chapter-level concurrency. DeepSeek and OpenAI embeddings both
# tolerate higher fan-out, but 8 keeps the error blast radius bounded
# and makes progress reporting readable.
_DEFAULT_WORKERS = 8

# Output cap for the extraction LLM call. V4-Flash supports up to 384K
# output tokens; we use 32K which comfortably fits phrase + paragraph
# extraction even on the longest chapters (part_229 has ~13K JP +
# ~38K EN = ~51K total chars; full bilingual extraction fits well
# under 32K output).
_EXTRACTION_MAX_TOKENS = 32768


# Path layout helpers ---------------------------------------------------------


def default_v2_root() -> Path:
    """Default location for the LLM-extracted index."""
    return VAULT_DIR / "precedent-index-v2"


def _granularity_paths(root: Path, granularity: str) -> tuple[Path, Path, Path]:
    base = root / granularity
    return (
        base / "pairs.jsonl",
        base / "embeddings.npy",
        base / "meta.json",
    )


def _cache_path(root: Path, part_id: str) -> Path:
    return root / "cache" / f"{part_id}.json"


# Data classes ----------------------------------------------------------------


@dataclass
class PhrasePair:
    part_id: str
    jp_span: str
    en_span: str
    category: str
    literal_alternative: str
    context_jp: str
    notes: str
    embedding: np.ndarray | None = None

    def to_record(self) -> dict:
        return {
            "part_id": self.part_id,
            "jp_span": self.jp_span,
            "en_span": self.en_span,
            "category": self.category,
            "literal_alternative": self.literal_alternative,
            "context_jp": self.context_jp,
            "notes": self.notes,
        }


@dataclass
class ParagraphPair:
    part_id: str
    jp: str
    en: str
    shape: str  # "1:1" | "1:2" | "2:1" | "2:2" | "N:M" generally
    embedding: np.ndarray | None = None

    def to_record(self) -> dict:
        return {
            "part_id": self.part_id,
            "jp_text": self.jp,
            "en_text": self.en,
            "shape": self.shape,
        }


@dataclass
class ChapterExtraction:
    """Raw extraction output for one chapter pair, after JSON parsing."""

    part_id: str
    phrase_pairs: list[PhrasePair] = field(default_factory=list)
    paragraph_pairs: list[ParagraphPair] = field(default_factory=list)


@dataclass
class ExtractionStats:
    parts_processed: list[str] = field(default_factory=list)
    parts_skipped_cached: list[str] = field(default_factory=list)
    parts_failed: list[str] = field(default_factory=list)
    phrase_count: int = 0
    paragraph_count: int = 0
    extraction_model: str = ""
    embedding_model: str = ""


# Prompt construction ---------------------------------------------------------

_SYSTEM_PROMPT = load_template("precedent_extract_system.md")

_USER_PROMPT_TEMPLATE = load_template("precedent_extract_user.md")


def _build_prompt(jp_text: str, en_text: str) -> str:
    return _USER_PROMPT_TEMPLATE.format(jp_text=jp_text, en_text=en_text)


# Response parsing ------------------------------------------------------------


def _parse_response(text: str, *, part_id: str) -> ChapterExtraction:
    """Robustly parse the LLM JSON output into a ChapterExtraction.

    DeepSeek occasionally wraps the JSON in markdown fences despite the
    prompt instruction; strip those before parsing.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse extraction JSON for {part_id}: {exc}\n"
            f"--- raw output (first 500 chars) ---\n{cleaned[:500]}"
        ) from exc

    phrase_pairs: list[PhrasePair] = []
    for pp in data.get("phrase_pairs", []):
        try:
            phrase_pairs.append(
                PhrasePair(
                    part_id=part_id,
                    jp_span=str(pp.get("jp_span", "")).strip(),
                    en_span=str(pp.get("en_span", "")).strip(),
                    category=str(pp.get("category", "other")).strip().lower(),
                    literal_alternative=str(
                        pp.get("literal_alternative", "")
                    ).strip(),
                    context_jp=str(pp.get("context_jp", "")).strip(),
                    notes=str(pp.get("notes", "")).strip(),
                )
            )
        except Exception:
            # One bad record shouldn't tank the whole chapter.
            continue

    paragraph_pairs: list[ParagraphPair] = []
    for pp in data.get("paragraph_pairs", []):
        try:
            paragraph_pairs.append(
                ParagraphPair(
                    part_id=part_id,
                    jp=str(pp.get("jp", "")).strip(),
                    en=str(pp.get("en", "")).strip(),
                    shape=str(pp.get("shape", "1:1")).strip(),
                )
            )
        except Exception:
            continue

    # Drop records with empty spans.
    phrase_pairs = [p for p in phrase_pairs if p.jp_span and p.en_span]
    paragraph_pairs = [p for p in paragraph_pairs if p.jp and p.en]

    return ChapterExtraction(
        part_id=part_id,
        phrase_pairs=phrase_pairs,
        paragraph_pairs=paragraph_pairs,
    )


# Quality validation ---------------------------------------------------------

# Caps mirror the prompt rules. Allow ~20% slack on the size caps so we
# don't reject borderline-good output (the model occasionally emits a
# pair just over 500/1500 chars but otherwise correctly broken up).
_MAX_PAIR_JP_CHARS = 600
_MAX_PAIR_EN_CHARS = 1800

# Coverage ratio: sum of indexed JP paragraph chars / source JP chars
# (whitespace stripped on both sides). 70% is a generous floor; clean
# extractions are typically 95-100%.
_MIN_PARA_COVERAGE = 0.70


class ExtractionQualityError(ValueError):
    """Extraction parsed but failed quality gates (mega-pair, low coverage,
    anemic count). Raised so the retry loop can try again with higher
    temperature / fresh sampling."""


def _validate_paragraph_quality(
    extraction: ChapterExtraction, jp_source: str
) -> None:
    """Raise ``ExtractionQualityError`` if paragraph_pairs look lazy.

    Three independent checks, any of which trips a retry:

    1. **Size cap**: no pair may exceed the per-pair JP/EN char caps.
    2. **Coverage**: total indexed JP chars must cover ≥ 70% of source.
    3. **Density**: for chapters with > 2000 source chars, must have
       ≥ 5 paragraph_pairs.
    """
    pairs = extraction.paragraph_pairs
    src_compact = "".join(jp_source.split())

    # Density check: is the chapter substantive but the count anemic?
    if len(src_compact) > 2000 and len(pairs) < 5:
        raise ExtractionQualityError(
            f"{extraction.part_id}: only {len(pairs)} paragraph_pair(s) for "
            f"a {len(src_compact)}-char chapter; LLM consolidated too much."
        )

    # Size-cap check: any individual pair too large?
    for i, p in enumerate(pairs):
        if len(p.jp) > _MAX_PAIR_JP_CHARS or len(p.en) > _MAX_PAIR_EN_CHARS:
            raise ExtractionQualityError(
                f"{extraction.part_id}: pair[{i}] is {len(p.jp)} JP / "
                f"{len(p.en)} EN chars — exceeds {_MAX_PAIR_JP_CHARS}/"
                f"{_MAX_PAIR_EN_CHARS} caps (mega-pair / catch-all)."
            )

    # Coverage check: did indexed pairs collectively span the source?
    indexed_compact_chars = sum(len("".join(p.jp.split())) for p in pairs)
    if src_compact:
        coverage = indexed_compact_chars / len(src_compact)
        if coverage < _MIN_PARA_COVERAGE:
            raise ExtractionQualityError(
                f"{extraction.part_id}: paragraph_pairs cover only "
                f"{coverage:.0%} of source ({indexed_compact_chars}/"
                f"{len(src_compact)} chars); LLM dropped content."
            )


# If after dropping non-verbatim phrases the chapter has fewer phrases
# than this floor (and the source is substantive), trigger retry.
_MIN_VERBATIM_PHRASE_COUNT = 5


def _normalize_for_verbatim(s: str) -> str:
    """Whitespace-stripped lowercase string for substring containment checks.

    The audit showed Flash often paraphrases punctuation/whitespace
    (curly→straight quotes, line breaks, capitalization). We accept all
    those normalizations as still-verbatim since the *content* is faithful.
    What we reject is content paraphrasing (different words).
    """
    return "".join(s.split()).lower()


def _filter_verbatim_phrases(
    extraction: ChapterExtraction, jp_source: str, en_source: str
) -> tuple[ChapterExtraction, int]:
    """Drop phrase pairs whose ``jp_span`` or ``en_span`` is not a verbatim
    substring of the source. Returns the filtered extraction and the
    number of dropped pairs.
    """
    jp_norm = _normalize_for_verbatim(jp_source)
    en_norm = _normalize_for_verbatim(en_source)

    kept: list[PhrasePair] = []
    for ph in extraction.phrase_pairs:
        jp_ok = _normalize_for_verbatim(ph.jp_span) in jp_norm
        en_ok = _normalize_for_verbatim(ph.en_span) in en_norm
        if jp_ok and en_ok:
            kept.append(ph)
    dropped = len(extraction.phrase_pairs) - len(kept)
    return (
        ChapterExtraction(
            part_id=extraction.part_id,
            phrase_pairs=kept,
            paragraph_pairs=extraction.paragraph_pairs,
        ),
        dropped,
    )


def _validate_phrase_yield(
    extraction: ChapterExtraction, jp_source: str
) -> None:
    """If verbatim filtering left too few phrases on a substantive
    chapter, trip a retry: the model paraphrased its way through and
    needs to try again with the verbatim rule reinforced by sampling
    variation."""
    src_compact = "".join(jp_source.split())
    if (
        len(src_compact) > 2000
        and len(extraction.phrase_pairs) < _MIN_VERBATIM_PHRASE_COUNT
    ):
        raise ExtractionQualityError(
            f"{extraction.part_id}: only {len(extraction.phrase_pairs)} "
            f"verbatim phrase(s) survived filtering on a "
            f"{len(src_compact)}-char chapter; model paraphrased too much."
        )


# Per-chapter extraction ------------------------------------------------------


def extract_chapter(
    part_id: str,
    *,
    model: str = DEFAULT_EXTRACTION_MODEL,
    cache_root: Path | None = None,
    force: bool = False,
) -> ChapterExtraction:
    """Extract precedents for a single chapter pair.

    Caches raw LLM output at ``cache_root/cache/{part_id}.json`` so
    re-running is free unless ``force=True``.
    """
    root = cache_root or default_v2_root()
    cp = _cache_path(root, part_id)

    jp_text = load_part_jp(part_id)
    en_text = load_part_en(part_id)
    if en_text is None:
        return ChapterExtraction(part_id=part_id)

    if not force and cp.exists():
        cached = json.loads(cp.read_text(encoding="utf-8"))
        candidate = _parse_response(cached["raw_response"], part_id=part_id)
        # Apply verbatim filtering on the cached response too, so cache
        # hits benefit from prompt+filter improvements without needing
        # a fresh LLM call.
        filtered, _ = _filter_verbatim_phrases(candidate, jp_text, en_text)
        return filtered

    prompt = _build_prompt(jp_text, en_text)

    # Three failure modes drive retry:
    #   1. JSON parse errors (unescaped quotes, stray control chars).
    #   2. Paragraph quality gates (mega-pair / lazy chapter coverage).
    #   3. Phrase paraphrasing (verbatim filter left too few survivors).
    # All three are non-deterministic — a fresh sample at a slightly
    # higher temperature usually succeeds. Try up to 4 attempts before
    # giving up.
    last_exc: Exception | None = None
    response = ""
    extraction: ChapterExtraction | None = None
    for attempt in range(4):
        # Stair-step temperature: 0.2 → 0.5 → 0.7 → 0.9. Higher samples
        # help the model produce a different segmentation rather than
        # the same lazy 1-pair answer.
        temp = [0.2, 0.5, 0.7, 0.9][attempt]
        response = complete(
            model=model,
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=_EXTRACTION_MAX_TOKENS,
            json_schema={"type": "object"},  # forces JSON mode on DeepSeek
        )
        try:
            candidate = _parse_response(response, part_id=part_id)
            _validate_paragraph_quality(candidate, jp_text)
            filtered, _dropped = _filter_verbatim_phrases(
                candidate, jp_text, en_text
            )
            _validate_phrase_yield(filtered, jp_text)
            extraction = filtered
            break
        except ValueError as exc:
            last_exc = exc
            continue
    if extraction is None:
        assert last_exc is not None
        raise last_exc

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        json.dumps(
            {
                "part_id": part_id,
                "model": model,
                "extracted_at": datetime.now(UTC).isoformat(),
                "raw_response": response,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return extraction


# Embedding -------------------------------------------------------------------


# OpenAI text-embedding-3-* models hard-cap each input at 8192 tokens.
# We truncate well below that so we always have margin and so the embed
# represents the *opening* of the paragraph (the most semantically
# distinctive portion); a runaway 30K-token paragraph_pair from the
# LLM extractor would otherwise crash the whole batch.
_EMBED_TOKEN_LIMIT = 7000


def _truncate_for_embedding(text: str, encoder) -> str:
    """Truncate ``text`` to ``_EMBED_TOKEN_LIMIT`` tokens using tiktoken.

    Returns the original string if it already fits. Decoding is needed
    because tiktoken's encode/decode round-trip is the only way to get
    a byte-correct truncation that respects multi-byte JP characters.
    """
    if not text:
        return text
    ids = encoder.encode(text)
    if len(ids) <= _EMBED_TOKEN_LIMIT:
        return text
    return encoder.decode(ids[:_EMBED_TOKEN_LIMIT])


def _embed_texts(
    texts: list[str],
    *,
    model: str,
    client=None,
) -> np.ndarray:
    """Embed and L2-normalize. Same convention as v1."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=require_openai_key())

    import tiktoken

    # text-embedding-3-* uses cl100k_base. tiktoken handles JP correctly.
    encoder = tiktoken.get_encoding("cl100k_base")

    rows: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        sanitized = [
            _truncate_for_embedding(t if t.strip() else " ", encoder)
            for t in batch
        ]
        resp = client.embeddings.create(model=model, input=sanitized)
        rows.extend(item.embedding for item in resp.data)

    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# Corpus-level orchestration --------------------------------------------------


def extract_corpus(
    *,
    parts: Iterable[str] | None = None,
    rebuild: bool = False,
    extraction_model: str = DEFAULT_EXTRACTION_MODEL,
    embedding_model: str | None = None,
    progress: bool = True,
    root: Path | None = None,
    workers: int = _DEFAULT_WORKERS,
) -> ExtractionStats:
    """Extract + embed precedents for the entire corpus (or a subset).

    Parameters
    ----------
    parts
        Specific part ids to process (e.g. ``["part_109", "part_181"]``).
        ``None`` = every chapter that has both JP and EN parallel files.
    rebuild
        If True, re-run the LLM on every chapter (ignore cache); otherwise
        cached outputs are reused.
    extraction_model
        Provider-routed model id. Defaults to ``deepseek-v4-flash``.
    embedding_model
        Embedding model for JP-side vectors. Defaults to ``MODELS.embedding``.
    progress
        Print per-chapter progress lines.
    root
        Override the index root. Defaults to ``knowledge-vault/precedent-index-v2``.
    workers
        Concurrent chapter-extraction workers. Bound by API rate limits.
    """
    embedding_model = embedding_model or MODELS.embedding
    root = root or default_v2_root()

    # Resolve which parts to process.
    if parts is None:
        candidate_ids = [
            part.id
            for part in iter_parts(only_translated=True, parts_only=True)
        ]
    else:
        candidate_ids = list(parts)

    eligible: list[str] = []
    for pid in candidate_ids:
        if load_part_en(pid) is None:
            continue
        eligible.append(pid)

    stats = ExtractionStats(
        extraction_model=extraction_model,
        embedding_model=embedding_model,
    )

    # Phase 1: per-chapter LLM extraction (cached, parallelized).
    extractions: list[ChapterExtraction] = []
    completed = 0

    def _run_one(pid: str) -> ChapterExtraction | Exception:
        try:
            cached = _cache_path(root, pid).exists() and not rebuild
            ex = extract_chapter(
                pid,
                model=extraction_model,
                cache_root=root,
                force=rebuild,
            )
            if cached:
                stats.parts_skipped_cached.append(pid)
            else:
                stats.parts_processed.append(pid)
            return ex
        except Exception as exc:  # pylint: disable=broad-except
            stats.parts_failed.append(pid)
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, pid): pid for pid in eligible}
        for fut in as_completed(futures):
            pid = futures[fut]
            result = fut.result()
            completed += 1
            if isinstance(result, Exception):
                if progress:
                    print(
                        f"[extract {completed}/{len(eligible)}] {pid}: "
                        f"FAILED — {type(result).__name__}: {result}"
                    )
                continue
            extractions.append(result)
            if progress:
                cached_marker = (
                    " (cached)" if pid in stats.parts_skipped_cached else ""
                )
                print(
                    f"[extract {completed}/{len(eligible)}] {pid}: "
                    f"{len(result.phrase_pairs)} phrase + "
                    f"{len(result.paragraph_pairs)} paragraph pair(s)"
                    f"{cached_marker}"
                )

    # Phase 2: flatten + embed JP sides.
    all_phrases: list[PhrasePair] = []
    all_paragraphs: list[ParagraphPair] = []
    for ex in extractions:
        all_phrases.extend(ex.phrase_pairs)
        all_paragraphs.extend(ex.paragraph_pairs)

    if progress:
        print(
            f"\n[embed] {len(all_phrases)} phrase + "
            f"{len(all_paragraphs)} paragraph spans → {embedding_model}"
        )

    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())

    if all_phrases:
        phrase_emb = _embed_texts(
            [p.jp_span for p in all_phrases],
            model=embedding_model,
            client=client,
        )
        for p, vec in zip(all_phrases, phrase_emb, strict=True):
            p.embedding = vec
    if all_paragraphs:
        para_emb = _embed_texts(
            [p.jp for p in all_paragraphs],
            model=embedding_model,
            client=client,
        )
        for p, vec in zip(all_paragraphs, para_emb, strict=True):
            p.embedding = vec

    # Phase 3: persist.
    _persist_granularity(
        granularity="phrases",
        records=[p.to_record() for p in all_phrases],
        embeddings=np.asarray(
            [p.embedding for p in all_phrases if p.embedding is not None],
            dtype=np.float32,
        )
        if all_phrases
        else np.zeros((0, 0), dtype=np.float32),
        meta={
            "granularity": "phrases",
            "extraction_model": extraction_model,
            "embedding_model": embedding_model,
            "count": len(all_phrases),
            "dim": int(all_phrases[0].embedding.shape[0])
            if all_phrases and all_phrases[0].embedding is not None
            else 0,
            "parts_indexed": sorted({p.part_id for p in all_phrases}),
            "last_updated": datetime.now(UTC).isoformat(),
        },
        root=root,
    )
    _persist_granularity(
        granularity="paragraphs",
        records=[p.to_record() for p in all_paragraphs],
        embeddings=np.asarray(
            [p.embedding for p in all_paragraphs if p.embedding is not None],
            dtype=np.float32,
        )
        if all_paragraphs
        else np.zeros((0, 0), dtype=np.float32),
        meta={
            "granularity": "paragraphs",
            "extraction_model": extraction_model,
            "embedding_model": embedding_model,
            "count": len(all_paragraphs),
            "dim": int(all_paragraphs[0].embedding.shape[0])
            if all_paragraphs and all_paragraphs[0].embedding is not None
            else 0,
            "parts_indexed": sorted({p.part_id for p in all_paragraphs}),
            "last_updated": datetime.now(UTC).isoformat(),
        },
        root=root,
    )

    # Top-level meta.
    (root / "meta.json").write_text(
        json.dumps(
            {
                "version": "v2",
                "extraction_model": extraction_model,
                "embedding_model": embedding_model,
                "phrase_count": len(all_phrases),
                "paragraph_count": len(all_paragraphs),
                "parts_processed": sorted(stats.parts_processed),
                "parts_skipped_cached": sorted(stats.parts_skipped_cached),
                "parts_failed": sorted(stats.parts_failed),
                "last_updated": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stats.phrase_count = len(all_phrases)
    stats.paragraph_count = len(all_paragraphs)
    return stats


def _persist_granularity(
    *,
    granularity: str,
    records: list[dict],
    embeddings: np.ndarray,
    meta: dict,
    root: Path,
) -> None:
    pairs_path, emb_path, meta_path = _granularity_paths(root, granularity)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    with pairs_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if embeddings.size:
        np.save(emb_path, embeddings)
    else:
        # Write an empty (0, 0) array so downstream load doesn't crash.
        np.save(emb_path, np.zeros((0, 0), dtype=np.float32))
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = [
    "DEFAULT_EXTRACTION_MODEL",
    "ChapterExtraction",
    "ExtractionStats",
    "ParagraphPair",
    "PhrasePair",
    "default_v2_root",
    "extract_chapter",
    "extract_corpus",
]
