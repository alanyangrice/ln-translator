"""Cluster a batch of deviation notes into candidate rules.

Reads:

* All deviation notes from the rounds whose IDs are passed in (typically
  the current round, but you can include earlier rounds for momentum).
* Active rules (so the clusterer doesn't regenerate them).
* Pruned rules with their reasons (so the clusterer doesn't re-propose
  them from the same evidence).

Writes the resulting candidate rules to ``rules/candidate/`` via
:mod:`translator.vault.rules`. The promotion / pruning step (Step 5 of
the v3 loop) is a separate concern handled by ``cli.evaluate report``.
"""

from __future__ import annotations

import json
from string import Template
from typing import Any

from translator.config import MODELS, PATHS, REASONING, VAULT
from translator.inference.provider import complete
from translator.vault import load_active_rules, load_pruned_rules, write_rule
from translator.vault.notes import list_notes, read_note
from translator.vault.rules import Rule
from translator.vault.templates import CLUSTERING_TEMPLATE


def _format_rules_for_prompt(rules: list[Rule]) -> str:
    if not rules:
        return "_(none)_"
    return "\n".join(
        f"- [{r.id}] {r.text} "
        f"(pov_scope={r.pov_scope}, scene_scope={r.scene_scope}, priority={r.priority}"
        f"{', prune_reason=' + r.prune_reason if r.prune_reason else ''})"
        for r in rules
    )


def _load_deviation_notes_for_rounds(round_numbers: list[int]) -> str:
    """Render every deviation note from the given rounds as one prompt block."""
    blocks: list[str] = []
    for round_number in round_numbers:
        round_dir = PATHS.knowledge_vault / VAULT.deviations / f"round-{round_number:02d}"
        for path in list_notes(round_dir):
            try:
                note = read_note(path)
            except (FileNotFoundError, ValueError):
                continue
            blocks.append(f"### {path.name}\n{note.body.strip()}")
    return "\n\n".join(blocks) if blocks else "_(no deviation notes found)_"


def _parse_candidate_rules(raw: str) -> list[dict[str, Any]]:
    """Parse the clustering model's JSON output. Tolerates code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if "```" in text:
            text = text.split("```", 1)[0]
    start = min((text.find(c) for c in "[{" if text.find(c) != -1), default=-1)
    if start > 0:
        text = text[start:]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("rules", [])
    return list(payload)


def cluster_into_candidate_rules(
    *,
    round_numbers: list[int],
    promote_round: int,
    model: str | None = None,
    write_to_vault: bool = True,
) -> list[Rule]:
    """Run the clustering model and return the candidate rules.

    ``round_numbers`` is the set of rounds whose deviation notes should
    feed the clusterer. ``promote_round`` is what gets stamped as the
    ``created_round`` in each new rule's frontmatter.
    """
    model = model or MODELS.clustering
    vault_template = PATHS.knowledge_vault / VAULT.clustering_template
    template_text = (
        vault_template.read_text(encoding="utf-8") if vault_template.exists() else CLUSTERING_TEMPLATE
    )
    template = Template(template_text)

    prompt = template.safe_substitute(
        active_rules=_format_rules_for_prompt(load_active_rules()),
        pruned_rules=_format_rules_for_prompt(load_pruned_rules()),
        deviations=_load_deviation_notes_for_rounds(round_numbers),
    )
    raw = complete(
        model=model,
        prompt=prompt,
        temperature=0.2,
        max_tokens=4096,
        reasoning_effort=REASONING.clustering,  # type: ignore[arg-type]
    )
    payload = _parse_candidate_rules(raw)

    rules: list[Rule] = []
    for i, item in enumerate(payload):
        rule_id = item.get("id") or f"rule-{promote_round:02d}-{i + 1:02d}"
        rule = Rule(
            id=rule_id,
            state="candidate",
            text=item.get("rule_text", "").strip(),
            pov_scope=list(item.get("pov_scope") or ["all"]),
            scene_scope=list(item.get("scene_scope") or ["all"]),
            priority=int(item.get("priority") or 0),
            created_round=promote_round,
            supporting_deviations=list(item.get("supporting_deviations") or []),
        )
        rules.append(rule)
        if write_to_vault and rule.text:
            write_rule(rule)
    return rules


__all__ = ["cluster_into_candidate_rules"]
