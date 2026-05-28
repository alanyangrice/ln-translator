"""JP↔EN paragraph alignment for the precedent index.

The corpus is paragraph-aligned by construction (one translator
paragraph per non-empty line on each side), but the human translator
sometimes splits one JP paragraph into multiple EN paragraphs or
merges two JP paragraphs into one. Across 229 chapters the JP/EN
paragraph-count ratio is 0.992 (median), with ~38% of chapters
showing drift larger than ±5 paragraphs — too much for a positional
zip but small enough that monotonic alignment captures it cleanly.

This module implements **Gale-Church style length-only DP
alignment**:

* Both sides are split into paragraphs (one per non-empty line).
* No EN embedding is ever computed — the cost function uses only
  character lengths and the corpus's measured EN/JP length ratio
  (``c ≈ 2.77`` chars-per-char, σ ≈ 0.24).
* A small DP enforces monotonicity and explicitly models 1:1, 1:2,
  2:1, 1:0, 0:1 alignment shapes. The result is a list of paragraph
  pairs, each of which becomes one precedent in the index.

The algorithm is intentionally embedding-free: alignment is a
sequence problem with strong positional and length priors, and we
don't want to pay for EN embeddings that get discarded after
alignment. The JP-side embedding still gets computed in ``index.py``
so retrieval can do JP→JP cosine search.

This module is pure: it just exposes ``split_jp_paragraphs``,
``split_en_paragraphs``, ``align_paragraphs``, and the resulting
dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Strip the leading full-width space some ``part_NNN.jp.txt`` files
# use as a paragraph indent. Keeping the indent in the embedded text
# makes every paragraph start with the same character and degrades
# JP→JP retrieval — this constant is removed before storing.
_JP_PARA_INDENT = "\u3000"

# Calibration constants for the Gale-Church length-DP cost.
#
# ``DEFAULT_LENGTH_RATIO`` is the empirical mean EN_chars / JP_chars
# across all 229 translated chapters (median 2.77, mean 2.82, σ 0.24
# *at the chapter level*).
#
# ``DEFAULT_LENGTH_VAR`` is the empirical *per-paragraph* residual
# variance s² ≈ (en − jp·c)² / jp_len, measured on chapters where JP
# and EN paragraph counts match exactly (1,557 paired paragraphs
# across 14 chapters, median 12.8). Per-paragraph variance is
# substantially larger than per-chapter ratio variance because the
# law of large numbers smooths the chapter average — the DP needs
# the per-paragraph value to keep 1:1 alignments competitive against
# skip transitions.
DEFAULT_LENGTH_RATIO = 2.77
DEFAULT_LENGTH_VAR = 13.0

# Allowed alignment "shapes" for the DP. Each is `(a, b, prior)`
# where ``a`` JP paragraphs align with ``b`` EN paragraphs and
# ``prior`` is the prior probability of the shape. We weight 1:1
# heavily because 78% of paragraphs are single-sentence dialogue or
# one-line narration that almost always aligns 1:1; merges and
# splits remain accessible but require the length-ratio evidence to
# overcome the prior.
_ALIGNMENT_SHAPES: tuple[tuple[int, int, float], ...] = (
    (1, 1, 0.92),  # one-to-one, the dominant case
    (1, 2, 0.04),  # JP paragraph split across two EN paragraphs
    (2, 1, 0.03),  # two JP paragraphs merged into one EN paragraph
    (1, 0, 0.005),  # JP paragraph dropped (very rare)
    (0, 1, 0.005),  # EN paragraph added (very rare)
)


@dataclass(frozen=True)
class JPParagraph:
    """One JP paragraph with its 0-based index in the chapter."""

    text: str
    index: int


@dataclass(frozen=True)
class ENParagraph:
    """One EN paragraph with its 0-based index in the chapter."""

    text: str
    index: int


@dataclass(frozen=True)
class ParagraphAlignment:
    """A monotonic JP↔EN paragraph alignment record.

    Both ``jp_paragraphs`` and ``en_paragraphs`` are tuples because a
    single record may cover 1:1, 1:2, or 2:1 — the DP emits one
    record per visited "node" in the alignment path.

    ``shape`` is ``(len(jp_paragraphs), len(en_paragraphs))``.
    ``length_score`` is in [0, 1] — higher is a closer length match.
    """

    jp_paragraphs: tuple[JPParagraph, ...]
    en_paragraphs: tuple[ENParagraph, ...]
    shape: tuple[int, int]
    length_score: float


def split_jp_paragraphs(text: str) -> list[JPParagraph]:
    """One JP paragraph per non-empty line, with the indent stripped."""
    out: list[JPParagraph] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip(_JP_PARA_INDENT).strip()
        if not stripped:
            continue
        out.append(JPParagraph(text=stripped, index=len(out)))
    return out


def split_en_paragraphs(text: str) -> list[ENParagraph]:
    """One EN paragraph per non-empty line."""
    out: list[ENParagraph] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        out.append(ENParagraph(text=stripped, index=len(out)))
    return out


def _length_log_prob(jp_chars: int, en_chars: int, c: float, var: float) -> float:
    """Gale-Church length log-probability.

    ``var`` is the per-character variance s² (so for a paragraph of
    ``jp_chars`` JP characters the predicted EN length has mean
    ``jp_chars * c`` and variance ``jp_chars * var``). The
    standardized residual squared is the negative log-likelihood up
    to a constant.

    For 1:0 and 0:1 (skip) shapes there's no length pairing — the
    cost is carried entirely by the prior, so we return 0.
    """
    if jp_chars == 0 or en_chars == 0:
        return 0.0
    expected = jp_chars * c
    variance = jp_chars * var
    if variance <= 0:
        return 0.0
    z = (en_chars - expected) / math.sqrt(variance)
    return -0.5 * z * z


def _length_score(jp_chars: int, en_chars: int, c: float, var: float) -> float:
    """Map the length log-probability into a 0..1 quality score for
    reporting. 1.0 == residual ~0; 0.0 == residual large.
    """
    if jp_chars == 0 and en_chars == 0:
        return 1.0
    if jp_chars == 0 or en_chars == 0:
        return 0.0
    expected = jp_chars * c
    variance = jp_chars * var
    if variance <= 0:
        return 0.0
    z = (en_chars - expected) / math.sqrt(variance)
    return float(math.exp(-0.5 * z * z))


def align_paragraphs(
    jp_paragraphs: list[JPParagraph],
    en_paragraphs: list[ENParagraph],
    *,
    length_ratio: float = DEFAULT_LENGTH_RATIO,
    length_var: float = DEFAULT_LENGTH_VAR,
) -> list[ParagraphAlignment]:
    """Monotonic length-based DP alignment.

    Returns a list of :class:`ParagraphAlignment` records covering
    every JP paragraph (a paragraph may participate in a 1:0 record
    if the translator dropped it, and EN paragraphs may participate
    in a 0:1 record if the translator added them).

    The DP table ``f[i][j]`` holds the maximum total log-probability
    for aligning ``jp[0:i]`` with ``en[0:j]``. We then backtrack to
    recover the alignment shapes, emitting one record per step. 1:0
    and 0:1 records are kept in the path (so the JP→EN coverage
    stays monotonic) but callers typically filter them out before
    indexing because they don't carry a useful precedent.
    """
    n = len(jp_paragraphs)
    m = len(en_paragraphs)
    if n == 0 and m == 0:
        return []
    jp_lens = [len(p.text) for p in jp_paragraphs]
    en_lens = [len(p.text) for p in en_paragraphs]

    neg_inf = -1e18
    # f[i][j] = best log-prob aligning jp[0:i] with en[0:j].
    f: list[list[float]] = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    # back[i][j] = (a, b) shape used to reach (i, j); None for the origin.
    back: list[list[tuple[int, int] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    f[0][0] = 0.0

    log_priors = {
        (a, b): math.log(prior) for a, b, prior in _ALIGNMENT_SHAPES
    }

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = neg_inf
            best_shape: tuple[int, int] | None = None
            for a, b in log_priors:
                pi, pj = i - a, j - b
                if pi < 0 or pj < 0:
                    continue
                jp_chunk = sum(jp_lens[pi:i])
                en_chunk = sum(en_lens[pj:j])
                length_lp = _length_log_prob(
                    jp_chunk, en_chunk, length_ratio, length_var
                )
                cand = f[pi][pj] + log_priors[(a, b)] + length_lp
                if cand > best:
                    best = cand
                    best_shape = (a, b)
            f[i][j] = best
            back[i][j] = best_shape

    # Backtrack from (n, m).
    path: list[tuple[int, int, int, int]] = []  # (i_start, i_end, j_start, j_end)
    i, j = n, m
    while i > 0 or j > 0:
        shape = back[i][j]
        if shape is None:
            # No path — shouldn't happen with non-zero priors but
            # guard anyway by emitting a 1:0 or 0:1 step.
            if i > 0:
                path.append((i - 1, i, j, j))
                i -= 1
            else:
                path.append((i, i, j - 1, j))
                j -= 1
            continue
        a, b = shape
        path.append((i - a, i, j - b, j))
        i -= a
        j -= b
    path.reverse()

    out: list[ParagraphAlignment] = []
    for i_lo, i_hi, j_lo, j_hi in path:
        jp_slice = tuple(jp_paragraphs[i_lo:i_hi])
        en_slice = tuple(en_paragraphs[j_lo:j_hi])
        jp_chars = sum(jp_lens[i_lo:i_hi])
        en_chars = sum(en_lens[j_lo:j_hi])
        score = _length_score(jp_chars, en_chars, length_ratio, length_var)
        out.append(
            ParagraphAlignment(
                jp_paragraphs=jp_slice,
                en_paragraphs=en_slice,
                shape=(i_hi - i_lo, j_hi - j_lo),
                length_score=score,
            )
        )
    return out


__all__ = [
    "DEFAULT_LENGTH_RATIO",
    "DEFAULT_LENGTH_VAR",
    "ENParagraph",
    "JPParagraph",
    "ParagraphAlignment",
    "align_paragraphs",
    "split_en_paragraphs",
    "split_jp_paragraphs",
]
