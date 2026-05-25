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
from translator.glossary import format_for_prompt as format_glossary_for_prompt
from translator.glossary import load_glossary
from translator.inference.provider import complete
from translator.vault import format_rules_for_prompt, load_active_rules

JUDGE_PROMPT = """\
You are a literary translation judge. Rate the candidate translation
against the human reference translation on four axes, each 1-5. Be
strict — a 4 means "publishable with light editing", a 5 means "as good
as the reference for this passage". Translationese, stiff rhythm, and
violations of active rules / glossary should drag the relevant axis
down even when the surface meaning is correct.

* semantic_accuracy — does the candidate convey the same content,
  events, and grammatical subjects as the reference? (Penalize subject
  flips, attribution swaps, fabricated specifics, and dropped beats.)
* voice_fidelity — does it match the established translator's voice for
  this POV — dry/observant for Sendai, terse/restrained for Miyagi —
  rather than a generic literal voice? Violations of voice/register
  rules count AGAINST this axis.
* naturalness — does the English read as native English written by a
  fluent novelist, with no calques, awkward noun-phrase choices, or
  other translationese? A sentence that is grammatically valid but
  that no native speaker would actually write counts AGAINST this
  axis. Violations of translationese-related rules count especially
  hard against this axis.
* style_match — does it imitate THIS specific reference translator's
  patterns — expansive sentences with subordinating clauses, frequent
  contractions, dialogue tags cushioned with concurrent action, beat-
  and-then paragraph structure — rather than just producing "valid
  English"? Violations of style-rhythm rules count AGAINST this axis.

# How to use the rules and glossary

The active rules and glossary below were given to the candidate
translator as ground truth. Use them as your standard for what
"correct" means: a candidate that violates an active rule or glossary
entry should not score above 3 on the corresponding axis. When you cite
a problem in a rationale, mention the rule ID (e.g. "violates
rule-000-06") so the rationale is actionable.

For each axis, write a one-sentence justification that calls out the
most representative example from the candidate text and, if relevant,
the rule it violates. Return ONLY a JSON object with these keys:

{{
  "semantic_accuracy": {{ "score": int, "rationale": str }},
  "voice_fidelity":    {{ "score": int, "rationale": str }},
  "naturalness":       {{ "score": int, "rationale": str }},
  "style_match":       {{ "score": int, "rationale": str }}
}}

# Materials

## Active rules
$active_rules

## Glossary
$glossary

## POV
$pov

## Japanese source
$jp

## Candidate translation
$candidate

## Reference translation
$reference
"""


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
    from string import Template

    model = model or MODELS.judge
    rules = load_active_rules()
    glossary_entries = load_glossary()
    prompt = Template(JUDGE_PROMPT).safe_substitute(
        pov=pov,
        jp=jp,
        candidate=candidate,
        reference=reference,
        active_rules=format_rules_for_prompt(rules),
        glossary=format_glossary_for_prompt(glossary_entries),
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
