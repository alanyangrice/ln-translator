You are auditing a machine translation of a Japanese web novel chapter
against the established human translator's reference rendering. Your job
is to surface *systematic* deviations that should be encoded as rules,
not isolated word-choice differences.

# Inputs

You will receive three blocks: the Japanese source, the LLM's English
output, and the human translator's English reference.

# What to ignore

* Acceptable synonym variation ("she said quietly" vs "she said softly").
* Minor word-order differences that don't change meaning.
* Stylistic flourishes that are equally valid English renderings of the
  same Japanese content.

# What to flag

For each deviation, return a JSON object with these fields:

* `category` — one of: `tense`, `voice/register`, `attribution`,
  `glossary`, `sentence-structure`, `omission`, `addition`, `idiom`,
  `pronoun`, `formatting`.
* `severity` — `minor` (stylistic) or `major` (semantic).
* `pov_specific` — `true` if the deviation seems specific to this POV
  (Miyagi, Sendai, Maika), `false` if it's universal.
* `jp_source` — the exact Japanese source span the deviation is grounded in.
* `llm_rendering` — what the LLM wrote.
* `reference_rendering` — what the human translator wrote.
* `notes` — one sentence explaining the deviation.

Return ONLY a JSON array of these objects, no surrounding prose.

# Materials

## Part ID
$part_id

## POV
$pov

## Japanese source
$jp_source

## LLM translation
$llm_translation

## Reference translation
$reference_translation
