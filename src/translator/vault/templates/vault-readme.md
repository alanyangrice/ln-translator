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
