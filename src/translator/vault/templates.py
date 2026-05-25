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

# RULES (learned from previous evaluations)

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
* `supporting_deviations` — array of deviation note IDs that motivated
  the rule.
* `rationale` — one sentence explaining why this is a pattern, not noise.

Return ONLY the JSON array.

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
