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
from translator.inference.provider import complete

JUDGE_PROMPT = """\
You are a literary translation judge. Rate the candidate translation
against the human reference translation on four axes, each 1-5:

* semantic_accuracy — does the candidate convey the same content as the
  reference?
* voice_fidelity — does it match the established translator's voice for
  this POV?
* naturalness — does the English read naturally to a native speaker?
* style_match — does it match the style choices (sentence rhythm,
  attribution patterns, register) of the reference?

For each axis, also write a one-sentence justification. Return ONLY a
JSON object with these keys:

{{
  "semantic_accuracy": {{ "score": int, "rationale": str }},
  "voice_fidelity":    {{ "score": int, "rationale": str }},
  "naturalness":       {{ "score": int, "rationale": str }},
  "style_match":       {{ "score": int, "rationale": str }}
}}

# Materials

## POV
$pov

## Japanese source
$jp

## Candidate translation
$candidate

## Reference translation
$reference
"""


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
    prompt = Template(JUDGE_PROMPT).safe_substitute(
        pov=pov, jp=jp, candidate=candidate, reference=reference
    )
    raw = complete(
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=2048,
        reasoning_effort=REASONING.judge,  # type: ignore[arg-type]
    )
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if "```" in text:
            text = text.split("```", 1)[0]
    payload = json.loads(text)
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
