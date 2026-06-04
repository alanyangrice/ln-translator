"""LLM-as-judge rubric scoring.

Per v3 the judge rates each translation on four axes:

* semantic accuracy
* character voice fidelity
* naturalness
* style match

The rubric returns a JSON object per chapter; the round summary
aggregates the means. This is a complement to COMET/BERTScore — those
catch surface drift, while the judge catches voice/style issues that
embed-similarity is blind to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from translator.config import MODELS, REASONING
from translator.eval.prompts import render
from translator.glossary import format_for_prompt as format_glossary_for_prompt
from translator.glossary import load_glossary
from translator.inference.provider import complete
from translator.style import format_style_profile_for_prompt
from translator.vault import format_rules_for_prompt, load_active_rules

_AXIS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

JUDGE_JSON_SCHEMA: dict = {
    "name": "judge_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "semantic_accuracy": _AXIS_SCHEMA,
            "voice_fidelity": _AXIS_SCHEMA,
            "naturalness": _AXIS_SCHEMA,
            "style_match": _AXIS_SCHEMA,
        },
        "required": [
            "semantic_accuracy",
            "voice_fidelity",
            "naturalness",
            "style_match",
        ],
        "additionalProperties": False,
    },
}


@dataclass
class JudgeResult:
    semantic_accuracy: int
    voice_fidelity: int
    naturalness: int
    style_match: int
    rationales: dict[str, str]
    raw: str

    @property
    def mean(self) -> float:
        return (
            self.semantic_accuracy
            + self.voice_fidelity
            + self.naturalness
            + self.style_match
        ) / 4.0


def judge_translation(
    *,
    pov: str,
    jp: str,
    candidate: str,
    reference: str,
    model: str | None = None,
) -> JudgeResult:
    """Call the judge model and return parsed scores."""
    model = model or MODELS.judge
    rules = load_active_rules()
    glossary_entries = load_glossary()
    prompt = render(
        "judge.md",
        pov=pov,
        jp=jp,
        candidate=candidate,
        reference=reference,
        active_rules=format_rules_for_prompt(rules),
        glossary=format_glossary_for_prompt(glossary_entries),
        style_profile=format_style_profile_for_prompt(),
    )
    raw = complete(
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=16384,  # reasoning tokens count toward this; high effort needs headroom
        reasoning_effort=REASONING.judge,  # type: ignore[arg-type]
        json_schema=JUDGE_JSON_SCHEMA,
    )
    payload = json.loads(raw)
    rationales: dict[str, str] = {}
    scores: dict[str, int] = {}
    for axis in ("semantic_accuracy", "voice_fidelity", "naturalness", "style_match"):
        item = payload.get(axis, {})
        scores[axis] = int(item.get("score", 0))
        rationales[axis] = str(item.get("rationale", ""))
    return JudgeResult(
        semantic_accuracy=scores["semantic_accuracy"],
        voice_fidelity=scores["voice_fidelity"],
        naturalness=scores["naturalness"],
        style_match=scores["style_match"],
        rationales=rationales,
        raw=raw,
    )


__all__ = ["JudgeResult", "judge_translation"]
