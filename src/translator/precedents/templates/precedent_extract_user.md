# Task

Extract translation precedents from the chapter pair below.

## phrase_pairs (high-precision idiom layer)

Identify JP phrases whose English rendering is **non-literal**: idioms, set expressions, facial-expression conventions, restructured grammar, condensed translations, culture-specific expressions. Skip phrases that translate 1:1 by dictionary substitution. Skip proper nouns and standard vocabulary. Be conservative — precision matters more than recall. Aim for 15-40 high-confidence pairs per chapter.

**Hard verbatim rule — violations cause the pair to be dropped:**

- ``jp_span`` MUST be a literal contiguous substring of the JP source. Copy the exact characters; do not paraphrase, normalize whitespace, or translate.
- ``en_span`` MUST be a literal contiguous substring of the human EN translation. Copy the exact characters; do not paraphrase, normalize punctuation, fix capitalization, or rephrase. If the rendering you want to capture spans multiple sentences, copy the full span verbatim including all punctuation between them.

If you cannot locate a verbatim ``en_span`` for a candidate idiom, **skip that candidate entirely**. A correctly-cited precedent is far more valuable than a paraphrased one.

For each pair:
- ``jp_span``: the exact JP substring as it appears in the source.
- ``en_span``: the exact EN substring from the human translation that renders this JP. Verbatim only — no paraphrasing or normalization.
- ``category``: one of ``facial_idiom``, ``set_phrase``, ``restructured``, ``condensed``, ``cultural``, ``other``.
- ``literal_alternative``: the stiff, word-for-word EN that a naive translator would produce (so the consumer can recognize the failure mode). This is the only field that may be invented; everything else must be cited.
- ``context_jp``: the full JP paragraph this span appears in (so downstream retrieval has surrounding context). Must be a verbatim substring of the source.
- ``notes``: one sentence on why the literal rendering would be wrong.

Examples of the *kind* of pair we want (not necessarily from this chapter):
- ``口角を上げて笑顔を作る`` → ``forced a smile`` (literal: "raised the corners of my mouth into a smile")
- ``予想の範囲内`` → ``I'd expected as much`` (literal: "within the range of expectation")
- ``息が止まりそうになる`` → ``took my breath away`` (literal: "my breath was about to stop")
- ``消化試合のようなもの`` → ``more like a formality`` (literal: "like a meaningless game")

## paragraph_pairs (voice/rhythm layer)

Map each JP paragraph to its EN counterpart and emit every pair in chapter order. The downstream consumer reconstructs the chapter by concatenating all pairs in sequence, so completeness and granularity are required.

**Hard rules — violations cause the extraction to be rejected:**

1. **Size cap**: every pair must contain ≤ 500 JP characters AND ≤ 1500 EN characters. If a passage exceeds this, split it into multiple consecutive pairs.
2. **Shape cap**: ``shape`` must be ``"a:b"`` where ``a`` and ``b`` are each in ``{{1, 2, 3}}``. Never emit ``5:5``, ``10:10``, ``33:33`` etc. — for higher fan-outs, emit several smaller consecutive pairs.
3. **No catch-all pairs**: never emit a single pair that absorbs the rest of the chapter. If you find yourself producing a pair with > 500 JP chars, you are doing it wrong — break it up.
4. **Exhaustive in order**: every JP paragraph must appear in exactly one pair, and pairs must appear in chapter order. Skip *only* JP paragraphs the translator deleted entirely, and *only* EN paragraphs the translator added with no JP source. Both should be rare.

For each pair:
- ``jp``: the JP paragraph(s), joined by ``\n`` if multi-paragraph. ≤ 500 chars.
- ``en``: the corresponding EN paragraph(s), joined by ``\n``. ≤ 1500 chars.
- ``shape``: ``"a:b"`` where ``a, b ∈ {{1, 2, 3}}``.

**Self-check before emitting**: count your paragraph_pairs. If the JP source has roughly N substantive paragraphs, you should have produced roughly N (give or take 30%) pairs. If you only have 1-3 pairs for a multi-paragraph chapter, you have failed rule #3 and #4 — go back and split.

## Output schema (strict JSON, no markdown, no commentary)

{{
  "phrase_pairs": [
    {{
      "jp_span": "...",
      "en_span": "...",
      "category": "facial_idiom",
      "literal_alternative": "...",
      "context_jp": "...",
      "notes": "..."
    }}
  ],
  "paragraph_pairs": [
    {{"jp": "...", "en": "...", "shape": "1:1"}}
  ]
}}

Important: emit valid JSON. Use double-quote escaping inside string values where needed. Do not wrap output in markdown code fences.

# Chapter pair

## JP source

{jp_text}

## EN translation (human, established)

{en_text}
