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
import re
from string import Template
from typing import Any

from translator.config import MODELS, PATHS, REASONING, VAULT
from translator.inference.provider import complete
from translator.vault import load_active_rules, load_pruned_rules, write_rule
from translator.vault.notes import list_notes, read_note
from translator.vault.rules import Rule, load_rules_by_state
from translator.vault.templates import CLUSTERING_TEMPLATE


# Deviation IDs are formed by `vault.deviations.write_deviation_note` as
# ``part-{chapter:03d}-r{round:02d}-d{idx}`` — extract the chapter number
# so we can name rules by the part they were learned from rather than the
# round they were proposed in.
_DEVIATION_ID_RE = re.compile(r"part-(\d{3})-r\d+-d\d+", re.ASCII)


def _chapter_from_supporting(supporting: list[str]) -> str:
    """Pick the primary chapter for a rule from its supporting deviations.

    Returns a zero-padded 3-digit string (e.g. ``"011"``). Falls back to
    ``"000"`` if the supporting list is missing or unparseable, which is
    harmless: the rule still writes, just under the ``rule-000-XX``
    namespace as a flag for human review.
    """
    for dev_id in supporting:
        match = _DEVIATION_ID_RE.match(dev_id.strip())
        if match:
            return match.group(1)
    return "000"


def _next_rule_index(chapter: str) -> int:
    """Return the next sequential rule number for ``chapter`` across all states.

    Scans candidate, active, pruned, and inactive — collisions across
    states would silently overwrite a rule's history if we ignored them.
    """
    pattern = re.compile(rf"rule-{chapter}-(\d+)$")
    used: set[int] = set()
    for state in ("candidate", "active", "pruned", "inactive"):
        for r in load_rules_by_state(state):
            m = pattern.match(r.id)
            if m:
                used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return n


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


CLUSTERING_JSON_SCHEMA: dict = {
    "name": "candidate_rules",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_text": {"type": "string"},
                        "pov_scope": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["miyagi", "sendai", "maika", "all"],
                            },
                        },
                        "scene_scope": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "dialogue",
                                    "internal_monologue",
                                    "action",
                                    "descriptive",
                                    "all",
                                ],
                            },
                        },
                        "priority": {"type": "integer"},
                        "supporting_deviations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "rule_text",
                        "pov_scope",
                        "scene_scope",
                        "priority",
                        "supporting_deviations",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rules"],
        "additionalProperties": False,
    },
}


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
        max_tokens=32768,
        reasoning_effort=REASONING.clustering,  # type: ignore[arg-type]
        json_schema=CLUSTERING_JSON_SCHEMA,
    )
    payload = _parse_candidate_rules(raw)

    rules: list[Rule] = []
    chapter_counters: dict[str, int] = {}
    for item in payload:
        supporting = list(item.get("supporting_deviations") or [])
        chapter = _chapter_from_supporting(supporting)
        if chapter not in chapter_counters:
            chapter_counters[chapter] = _next_rule_index(chapter)
        seq = chapter_counters[chapter]
        chapter_counters[chapter] += 1
        rule_id = item.get("id") or f"rule-{chapter}-{seq:02d}"
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
