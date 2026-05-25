"""Canonical templates written into ``knowledge-vault/config/`` on init.

These are the source-of-truth strings for the translation prompt and the
deviation-comparison prompt. Once the vault is initialized, the on-disk
copies are what ``inference.prompt`` actually loads — edit them in the
vault to tune behavior, and the pipeline picks up the change on the
next run. Re-running ``vault init`` does *not* overwrite existing
templates (so user edits survive).

The templates use ``string.Template`` ``$placeholder`` syntax so the
files render as valid Markdown in Obsidian and don't require Jinja.
"""

from __future__ import annotations

PROMPT_TEMPLATE = """\
You are translating the Japanese web novel *Sendai-san Always Says
Such Unnecessary Things* into English.

Below are recently translated chapters shown as Japanese-English pairs.
Translate the new chapter at the end to match this translator's voice,
style, and conventions exactly.

# STYLE PROFILE (target — match this prose signature)

The following profile characterizes the established human translator's
prose along 16 dimensions. Treat it as the *target* your output should
sound like, not a checklist to mechanically satisfy. The reference
chapters below are the ground truth; the profile is a summary of what
makes them sound the way they do.

$style_profile

# RULES (always follow these)

$rules

# GLOSSARY (always follow these)

$glossary

---

$reference_parts

---

# NEW CHAPTER TO TRANSLATE [$new_part_id]

$new_jp_chapter

# TRANSLATION
"""


COMPARISON_TEMPLATE = """\
You are auditing a machine translation of a Japanese web novel chapter
against the established human translator's reference rendering. Your job
is to surface *systematic* deviations that should drive rule promotion,
rule pruning, and glossary updates — not isolated word-choice differences.

# Inputs

You will receive:

* The current set of **active rules** the LLM was supposed to follow.
* The current **glossary** (hard-constraint term mappings the LLM was
  supposed to use).
* The current **style profile** characterizing the reference
  translator's prose along 16 dimensions.
* The Japanese source.
* The LLM's English output.
* The human translator's English reference.

# How to use the rules, glossary, and style profile

* **Treat the active rules as ground truth for what "correct" means.**
  If the LLM's rendering violates a rule, flag it with that rule's ID
  in the `violates_rule_id` field. Rule violations are the most
  actionable signal — they tell us a rule isn't working.
* **Treat glossary entries the same way.** If the LLM used a term that
  contradicts a glossary mapping (e.g. it wrote "muffler" when the
  glossary says マフラー → "scarf"), flag with `category: glossary`.
* **Treat the style profile as the target signature** the LLM was
  supposed to imitate. When the LLM output diverges from a profile
  dimension that the reference clearly demonstrates (e.g. profile says
  "frequent contractions in narration" but the LLM uses formal forms),
  flag with `category: style-profile`. Cite the dimension number in
  `notes` (e.g. "violates style profile §3 sentence structure").
* **Don't propose new rules that duplicate existing active or pruned
  rules.** Just flag the violation; the clustering step decides whether
  to update the existing rule.

# What to ignore

* Genuine synonym variation **where both renderings are natural native
  English** ("she said quietly" vs "she said softly"). Calques and
  unnatural noun-phrase choices are NOT synonyms — flag those as
  `translationese` even if the literal meaning matches.
* Minor word-order differences that don't change meaning *and* don't
  change rhythm noticeably.

# What to flag

For each deviation, return a JSON object with these fields:

* `category` — one of: `tense`, `voice/register`, `attribution`,
  `glossary`, `sentence-structure`, `omission`, `addition`, `idiom`,
  `pronoun`, `formatting`, `translationese`, `style-rhythm`,
  `style-profile`.
* `severity` — `minor` (stylistic) or `major` (semantic).
* `pov_specific` — `true` if the deviation seems specific to this POV
  (Miyagi, Sendai, Maika), `false` if it's universal.
* `jp_source` — the exact Japanese source span the deviation is grounded in.
* `llm_rendering` — what the LLM wrote.
* `reference_rendering` — what the human translator wrote.
* `notes` — one sentence explaining the deviation.
* `violates_rule_id` — the ID of the active rule this deviation
  violates (e.g. `rule-000-06`), or empty string if no active rule
  applies. Pick the *most specific* rule when multiple could match.

Return a JSON object of the form `{"deviations": [...]}` containing the
deviations array; no surrounding prose.

# Materials

## Active rules
$active_rules

## Glossary
$glossary

## Style profile
$style_profile

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
"""


CRITIQUE_TEMPLATE = """\
You are an inline critic auditing a draft machine translation of a
Japanese web novel chapter. You do NOT have the human reference; judge
the draft purely on whether it reads as native, fluent English written
by a literary novelist whose prose matches the established translator's
signature.

Your output drives a *revision pass*: only flag spans where a concrete
rewrite is available and the revised version would be a clear
improvement. Over-flagging trivial style preferences wastes a paid
revision round; under-flagging leaves translationese in the final
output. Aim for the calques, voice mismatches, and style-profile
divergences that a careful human editor would circle.

# Inputs

You will receive:

* The current set of **active rules** the draft was supposed to follow.
* The current **glossary** (hard-constraint term mappings).
* The **style profile** characterizing the reference translator's prose
  along 16 dimensions.
* The Japanese source.
* The draft English translation to audit.

# How to use the rules, glossary, and style profile

* **Active rules are ground truth.** If the draft violates a rule, flag
  it and cite the rule ID in `violates_rule_id`.
* **Glossary entries are hard constraints.** Flag any draft term that
  contradicts a glossary mapping.
* **Style profile is the target signature.** When the draft diverges
  from a profile dimension that the reference clearly demonstrates
  (e.g. profile says "frequent contractions" but the draft uses formal
  forms), flag with `category: style-profile` and cite the dimension
  in `notes` (e.g. "violates style profile §3 sentence structure").

# What to flag

Focus on:

* **Translationese / calques** (most important). Phrases that are
  grammatically valid English but that mirror Japanese surface
  structure rather than what an English novelist would write. Common
  patterns: literal verb-object pairings (`手をかざす` → "held my hand
  against"), unnatural noun phrases (`カットソー` → "cut-and-sew tee"),
  noun phrases mirroring JP modifier ordering, mechanical idiom
  literalizations (`喉を潤す` → "quench my throat"). When in doubt,
  read the span aloud — if a native speaker would think "this sounds
  translated," flag it.
* **Voice / register mismatches** for the POV character (Sendai,
  Miyagi, Maika).
* **Style-profile divergences** — sentence rhythm, paragraph
  structure, dialogue cushioning, contractions.
* **Glossary or active-rule violations**.

Skip:

* Genuine synonym variation where both options would read naturally.
* Word-order differences that don't change meaning or rhythm.
* Trivial nits when the surrounding prose is already strong.

# Output format

For each issue, return a JSON object with these fields:

* `category` — one of: `tense`, `voice/register`, `attribution`,
  `glossary`, `sentence-structure`, `omission`, `addition`, `idiom`,
  `pronoun`, `formatting`, `translationese`, `style-rhythm`,
  `style-profile`.
* `severity` — `major` (clearly translated / breaks voice / loses
  meaning) or `minor` (mildly stiff / locally awkward).
* `pov_specific` — `true` if the issue is specific to this POV.
* `jp_source` — the JP span the issue is grounded in, or empty
  string for pure-English rhythm/voice flags.
* `span` — the exact substring of the **draft** that needs replacing.
  Must appear verbatim in the draft so the revision pass can locate
  it.
* `suggested_rewrite` — a concrete English rewrite of `span`, written
  in the reference translator's voice. Self-contained: a careful editor
  could substitute it in directly.
* `notes` — one-sentence rationale (and rule/dimension citation if
  applicable).
* `violates_rule_id` — rule ID if applicable, else empty string.

Return a JSON object of the form `{"flags": [...]}` containing the
flags array; no surrounding prose.

# Materials

## Active rules
$active_rules

## Glossary
$glossary

## Style profile
$style_profile

## Part ID
$part_id

## POV
$pov

## Japanese source
$jp_source

## Draft translation
$draft
"""


REVISE_TEMPLATE = """\
You are translating the Japanese web novel *Sendai-san Always Says
Such Unnecessary Things* into English.

This is a **revision pass**. A separate critic flagged the spans below
in your previous draft as reading translated rather than as native
English, or as diverging from the reference translator's signature.
Produce a revised translation that:

1. **Applies all flagged fixes verbatim or with equivalent rewrites
   that resolve the same issue.** Do not retain the flagged spans.
2. **Improves analogous sentences nearby**, even if not flagged.
   The critic surfaces specific instances of patterns; the same
   pattern often shows up elsewhere in the chapter.
3. **Keeps unflagged content** as-is unless changing it improves
   coherence with the fixed passages. Don't rewrite for the sake of
   rewriting.

Output the full revised chapter — not a diff, not just the patched
sections. Match the formatting of the previous draft (paragraphs,
brackets, dialogue layout) except where the fix requires a change.

# STYLE PROFILE (target — match this prose signature)

$style_profile

# RULES (always follow these)

$rules

# GLOSSARY (always follow these)

$glossary

---

$reference_parts

---

# JAPANESE SOURCE [$new_part_id]

$new_jp_chapter

# CRITIC FLAGS (apply these fixes)

$critic_flags

# PREVIOUS DRAFT (revise from this)

$previous_draft

# REVISED TRANSLATION
"""


CLUSTERING_TEMPLATE = """\
You are turning a batch of per-chapter deviation notes into candidate
translation rules for a sliding-window LLM translator.

# Inputs

* All deviation notes from this batch, each tagged with round number
  and POV.
* The current set of *active* rules (do not regenerate these).
* The current set of *pruned* rules with their pruning rationale (do not
  regenerate these unless you have new evidence that addresses why they
  were pruned).

# What to produce

A JSON array of candidate rule objects, each with:

* `rule_text` — a single imperative sentence (or short paragraph) the
  translator should follow.
* `pov_scope` — array, any of: `miyagi`, `sendai`, `maika`, `all`.
* `scene_scope` — array, any of: `dialogue`, `internal_monologue`,
  `action`, `descriptive`, `all`.
* `priority` — integer; higher = wins when two rules conflict in the
  same context.
* `supporting_deviations` — array of deviation IDs that motivated the
  rule. **Use the IDs verbatim from the deviation note headers**, which
  follow the format `part-XXX-rNN-dNN` (e.g. `part-012-r01-d03`). Do
  not invent IDs; do not paraphrase. The chapter number in the ID is
  used to name the resulting candidate rule, so accuracy matters.
* `rationale` — one sentence explaining why this is a pattern, not noise.

Return a JSON object of the form `{"rules": [...]}` containing the
candidate rule objects; no surrounding prose.

# Materials

## Active rules
$active_rules

## Pruned rules (with reasons)
$pruned_rules

## Deviations from this batch
$deviations
"""


GLOSSARY_SCAFFOLD = """\
# Glossary

Hard-constraint term choices the translator must follow. Curated from
the parallel corpus and refined as evaluation surfaces inconsistencies.

| Japanese | English | Notes |
|----------|---------|-------|
| マフラー | scarf | Not "muffler" |
| 共用スペース | living room | Shared apartment space |
| 仙台さん | Sendai-san | Always with honorific (Miyagi POV) |
| 宮城 | Miyagi | No honorific (Sendai POV) |
| ピアス | earrings | — |
| 「」 | 「」 | Preserve Japanese dialogue brackets |
"""


STYLE_README = """\
# Style profile

The reference translator's prose signature, characterized along 16
dimensions. Each dimension lives in its own Markdown file in this
directory (`01-tone.md`, `02-voice.md`, …) so it can be hand-edited,
linked, or overridden independently. At prompt time the translator
concatenates all dimension bodies in numerical order and injects them
into the translation, comparison, and judge prompts via the
`$style_profile` placeholder.

## Bootstrap

```
translator style extract --through part_050
```

Reads the EN reference corpus through the named part (minus holdout
members), runs the extraction model, and writes one file per
dimension. Re-running overwrites all 16 files.

## Dimensions

1. Tone
2. Voice
3. Sentence structure
4. Word choice
5. Narrative distance
6. Reader trust
7. Internal monologue style
8. Pacing
9. Figurative language
10. Dialogue integration
11. Paragraph structure
12. Repetition and motif
13. Sensory emphasis
14. Tense and temporal framing
15. Connective tissue
16. Character voice differentiation

Until extraction has been run, the prompts will inject a placeholder
notice and translation continues normally.
"""


VAULT_README = """\
# knowledge-vault

This vault is the persistent knowledge store for the
[ln-translator](https://github.com/) translation pipeline. It holds:

* `glossary/` — hard-constraint term mappings
* `rules/{active,candidate,pruned,inactive}/` — translation rules at
  each stage of the lifecycle
* `deviations/round-NN/` — per-chapter deviation notes
* `evaluations/round-NN-summary.md` — round-level summary of metrics
  and rule changes
* `config/{prompt-template,comparison-prompt-template}.md` — the
  canonical prompts the pipeline loads at runtime

This is a normal Obsidian vault: open the folder in Obsidian to browse
notes, follow backlinks between rules and the deviations that
motivated them, and edit prose by hand if needed.

The vault is intended to be tracked in git. Each round of the
self-evaluation loop should land as one commit so any historical
translation can be reproduced by checking out the corresponding
commit.
"""
