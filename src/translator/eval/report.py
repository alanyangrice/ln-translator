"""Round summary writer.

Aggregates the round's scores, rule changes, and validation status into
a single Markdown note at ``evaluations/round-NN-summary.md`` so the
vault has a human-readable record of every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from translator.config import PATHS, VAULT
from translator.vault.notes import Note, write_note


@dataclass
class RoundSummary:
    round_number: int
    chapters_evaluated: int
    comet_mean: float | None = None
    bertscore_mean: float | None = None
    judge_means: dict[str, float] = field(default_factory=dict)
    rules_added: int = 0
    rules_pruned: int = 0
    rules_inactive: int = 0
    rules_active_total: int = 0
    notes: str = ""

    def to_note(self) -> Note:
        meta: dict[str, object] = {
            "round": self.round_number,
            "chapters_evaluated": self.chapters_evaluated,
            "rules_added": self.rules_added,
            "rules_pruned": self.rules_pruned,
            "rules_inactive": self.rules_inactive,
            "rules_active_total": self.rules_active_total,
        }
        if self.comet_mean is not None:
            meta["comet_mean"] = self.comet_mean
        if self.bertscore_mean is not None:
            meta["bertscore_mean"] = self.bertscore_mean
        if self.judge_means:
            meta["judge_means"] = self.judge_means

        lines = [
            f"# Round {self.round_number:02d} Summary",
            "",
            f"- **Chapters evaluated:** {self.chapters_evaluated}",
        ]
        if self.comet_mean is not None:
            lines.append(f"- **COMET mean:** {self.comet_mean:.4f}")
        if self.bertscore_mean is not None:
            lines.append(f"- **BERTScore mean:** {self.bertscore_mean:.4f}")
        if self.judge_means:
            lines.append("- **Judge means:**")
            for axis, value in self.judge_means.items():
                lines.append(f"  - {axis}: {value:.2f}")
        lines.extend(
            [
                "",
                "## Rule changes",
                "",
                f"- Added (candidate -> active): {self.rules_added}",
                f"- Pruned (regressed): {self.rules_pruned}",
                f"- Inactive (no measurable effect): {self.rules_inactive}",
                f"- Active rules total: {self.rules_active_total}",
            ]
        )
        if self.notes.strip():
            lines.extend(["", "## Notes", "", self.notes.strip()])
        return Note(meta=meta, body="\n".join(lines))


def write_round_summary(summary: RoundSummary, root: Path | None = None) -> Path:
    root = root or PATHS.knowledge_vault
    target = root / VAULT.evaluations / f"round-{summary.round_number:02d}-summary.md"
    write_note(target, summary.to_note())
    return target


__all__ = ["RoundSummary", "write_round_summary"]
