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
