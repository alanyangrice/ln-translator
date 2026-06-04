# Task

Scan the Japanese chapter below and identify phrases at risk of producing translationese.

## What to flag

A phrase is "at risk" if a literal word-for-word rendering would be stilted, unidiomatic, or technically correct but unnatural English. The categories (same vocabulary as our precedent index):

- ``facial_idiom``: 口角を上げる / 眉をひそめる / 目を細める / etc. where the literal physical description is wrong; English uses verbs like "smiled", "frowned", "squinted".
- ``set_phrase``: stock JP phrases (予想の範囲内, 言うまでもなく, 無難に終わる, 当たり前 etc.) that have clean idiomatic English equivalents.
- ``restructured``: phrases where natural English requires restructuring the grammar (subject changes, voice flip, condensed clauses).
- ``condensed``: JP phrases that English natively expresses in fewer words / one verb (余裕だった → "called it easy"; よくわからない → "can't tell").
- ``cultural``: culture-specific terms (制服, 部活, 先輩, family-name + さん addressing, etc.) where context determines whether to romanize, calque, or paraphrase.
- relationship / contract motifs: recurring story terms where a loose English synonym changes the relationship logic (命令 vs. お願い / リクエスト, いうことをきく, 私のもの, バイト). Flag these when the chapter context makes the distinction important, even if the word is not an idiom.
- ``other``: any other phrase you'd flag from experience.

Skip phrases that translate cleanly word-for-word. Skip standard vocabulary, proper nouns, and generic dialogue. Be precise — the downstream consumer is the translator who will look up established precedents for each flagged span.

For each entry:
- ``jp_span``: the exact JP substring (verbatim from the source).
- ``category``: one of the categories above.
- ``risk_level``: ``"high"`` (this WILL produce translationese unless addressed), ``"medium"`` (likely but a careful translator could land it), ``"low"`` (worth flagging but minor).
- ``literal_trap``: the stiff word-for-word EN that an inexperienced translator would produce. Be concrete.
- ``reason``: one sentence explaining why the literal is wrong.
- ``context_jp``: the full JP paragraph this span appears in (verbatim, so retrieval can use the surrounding context).

Be exhaustive within the category boundaries. A 2000-3000 char chapter typically yields 25-60 flagged phrases. A risk-light chapter of pure dialogue might yield fewer.

## Output schema (strict JSON, no markdown, no commentary)

{{
  "risks": [
    {{
      "jp_span": "...",
      "category": "facial_idiom",
      "risk_level": "high",
      "literal_trap": "...",
      "reason": "...",
      "context_jp": "..."
    }}
  ]
}}

# Chapter

{jp_text}
