"""Rule lifecycle helpers for the knowledge-vault.

Rules move through four states:

    candidate -> active   (validation showed score improvement)
              -> pruned   (validation showed score regression)
              -> inactive (no measurable effect; reconsider next round)

Pruned rules stay in the vault as institutional memory: the clustering
LLM reads them with their pruning rationale so it doesn't regenerate
them from the same evidence.

Each rule lives at ``rules/{state}/{rule_id}.md`` with frontmatter that
captures scope (POV, scene type), priority, supporting deviations, and
score deltas from the round that promoted/pruned it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from translator.config import PATHS, VAULT
from translator.vault.notes import Note, list_notes, read_note, write_note

RuleState = Literal["active", "candidate", "pruned", "inactive"]

POVScope = Literal["miyagi", "sendai", "maika", "all"]
SceneScope = Literal["dialogue", "internal_monologue", "action", "descriptive", "all"]


_STATE_TO_SUBDIR: dict[RuleState, str] = {
    "active": VAULT.rules_active,
    "candidate": VAULT.rules_candidate,
    "pruned": VAULT.rules_pruned,
    "inactive": VAULT.rules_inactive,
}


@dataclass
class Rule:
    """A single translation rule, materialized from a vault note."""

    id: str
    state: RuleState
    text: str
    pov_scope: list[POVScope] = field(default_factory=lambda: ["all"])
    scene_scope: list[SceneScope] = field(default_factory=lambda: ["all"])
    priority: int = 0
    created_round: int | None = None
    last_validated_round: int | None = None
    supporting_deviations: list[str] = field(default_factory=list)
    score_delta: float | None = None
    prune_reason: str | None = None
    body_extra: str = ""  # examples, backlinks, anything below the rule text
    path: Path | None = None

    def to_note(self) -> Note:
        body = f"# Rule\n\n{self.text.strip()}\n"
        if self.body_extra.strip():
            body += "\n" + self.body_extra.strip() + "\n"
        meta: dict[str, object] = {
            "id": self.id,
            "state": self.state,
            "pov_scope": list(self.pov_scope),
            "scene_scope": list(self.scene_scope),
            "priority": self.priority,
            "supporting_deviations": list(self.supporting_deviations),
        }
        if self.created_round is not None:
            meta["created_round"] = self.created_round
        if self.last_validated_round is not None:
            meta["last_validated_round"] = self.last_validated_round
        if self.score_delta is not None:
            meta["score_delta"] = self.score_delta
        if self.prune_reason is not None:
            meta["prune_reason"] = self.prune_reason
        return Note(meta=meta, body=body)

    @classmethod
    def from_note(cls, note: Note) -> Rule:
        meta = note.meta
        rule_id = str(meta.get("id") or (note.path.stem if note.path else ""))
        if not rule_id:
            raise ValueError(f"Rule note missing 'id' frontmatter: {note.path}")
        body = note.body.strip()
        # Strip the leading "# Rule\n\n" header if present, then split off
        # the first paragraph as the rule text and keep the rest as body_extra.
        if body.startswith("# Rule"):
            body = body.split("\n", 1)[1].strip() if "\n" in body else ""
        if "\n\n" in body:
            text, body_extra = body.split("\n\n", 1)
        else:
            text, body_extra = body, ""
        return cls(
            id=rule_id,
            state=meta.get("state", "candidate"),
            text=text.strip(),
            pov_scope=list(meta.get("pov_scope") or ["all"]),
            scene_scope=list(meta.get("scene_scope") or ["all"]),
            priority=int(meta.get("priority") or 0),
            created_round=meta.get("created_round"),
            last_validated_round=meta.get("last_validated_round"),
            supporting_deviations=list(meta.get("supporting_deviations") or []),
            score_delta=meta.get("score_delta"),
            prune_reason=meta.get("prune_reason"),
            body_extra=body_extra.strip(),
            path=note.path,
        )


def _state_dir(state: RuleState, root: Path | None = None) -> Path:
    root = root or PATHS.knowledge_vault
    return root / _STATE_TO_SUBDIR[state]


def load_rules_by_state(state: RuleState, root: Path | None = None) -> list[Rule]:
    """Load every rule currently in the given state. Returns ``[]`` if the
    vault doesn't exist or the state directory is empty."""
    rules: list[Rule] = []
    for path in list_notes(_state_dir(state, root)):
        try:
            rules.append(Rule.from_note(read_note(path)))
        except (ValueError, FileNotFoundError):
            continue
    rules.sort(key=lambda r: (-r.priority, r.id))
    return rules


def load_active_rules(root: Path | None = None) -> list[Rule]:
    return load_rules_by_state("active", root)


def load_pruned_rules(root: Path | None = None) -> list[Rule]:
    return load_rules_by_state("pruned", root)


def write_rule(rule: Rule, root: Path | None = None) -> Path:
    """Write a rule to the directory matching its state. Returns the path."""
    target = _state_dir(rule.state, root) / f"{rule.id}.md"
    write_note(target, rule.to_note())
    return target


def format_rules_for_prompt(rules: list[Rule]) -> str:
    """Render a list of rules as a numbered Markdown list for prompt injection.

    Used by the translation prompt, the deviation auditor, and the LLM
    judge so all three see the same rule text — the single source of
    truth lives in ``rules/active/*.md`` and only this helper knows how
    to format them.

    Each rule renders as a numbered block: imperative text on the first
    line with optional scope tags, and any ``body_extra`` (typically
    ``## Examples`` sections from the vault note) indented underneath.
    Empty rule list yields an explicit "no rules yet" line so consumers
    can interpolate without checking for emptiness.
    """
    if not rules:
        return "_(no rules yet — first round of self-evaluation hasn't completed)_"
    blocks: list[str] = []
    for n, r in enumerate(rules, start=1):
        scope_bits: list[str] = []
        if r.pov_scope and r.pov_scope != ["all"]:
            scope_bits.append(f"POV: {', '.join(r.pov_scope)}")
        if r.scene_scope and r.scene_scope != ["all"]:
            scope_bits.append(f"scene: {', '.join(r.scene_scope)}")
        scope = f" _({'; '.join(scope_bits)})_" if scope_bits else ""
        block = f"{n}. **[{r.id}]** {r.text.strip()}{scope}"
        extra = r.body_extra.strip()
        if extra:
            indented = "\n".join("   " + line if line else "" for line in extra.splitlines())
            block += "\n\n" + indented
        blocks.append(block)
    return "\n\n".join(blocks)


def promote_rule(
    rule_id: str,
    *,
    from_state: RuleState,
    to_state: RuleState,
    score_delta: float | None = None,
    prune_reason: str | None = None,
    last_validated_round: int | None = None,
    root: Path | None = None,
) -> Path:
    """Move a rule from one state to another, updating provenance metadata.

    The old file is removed; the new file is written under the target
    state's directory. Raises ``FileNotFoundError`` if the source rule
    can't be located.
    """
    src = _state_dir(from_state, root) / f"{rule_id}.md"
    if not src.exists():
        raise FileNotFoundError(src)
    rule = Rule.from_note(read_note(src))
    rule.state = to_state
    if score_delta is not None:
        rule.score_delta = score_delta
    if prune_reason is not None and to_state == "pruned":
        rule.prune_reason = prune_reason
    if last_validated_round is not None:
        rule.last_validated_round = last_validated_round
    new_path = write_rule(rule, root)
    src.unlink()
    return new_path
