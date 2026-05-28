"""LLM-driven at-risk scanner for new chapters.

Counterpart to :mod:`translator.precedents.extract`. Where ``extract``
runs over the parallel corpus to mine JP→EN precedents, ``risk`` runs
over a *new* (untranslated) chapter to flag the JP phrases most likely
to produce translationese unless the translator reaches for an
established idiomatic rendering. The output feeds into
:mod:`translator.precedents.retrieve` so we can do *targeted* phrase
retrieval against the v2 phrase index, instead of relying solely on
paragraph-level cosine similarity (which dilutes idiom signals).

Why this layer exists
---------------------
Manual review of ``part_230`` showed paragraph-level retrieval
consistently misses short idiomatic JP phrases (``口角を上げて笑顔を作る``
→ ``forced a smile`` etc.) because the idiom is one short fragment
inside a 200-character paragraph; the cosine similarity is dominated
by the surrounding context. A second pass that *names* the at-risk
fragments lets us retrieve precedents at the phrase granularity where
the signal is concentrated.

Design
------
* Sonnet 4.6 by default — good Japanese / literary judgment at lower
  cost than Opus 4.7. DeepSeek V4-Pro is also a fine substitute (same
  family that built the corpus, so its category vocabulary aligns).
* Strict JSON output, identical category vocabulary as ``extract.py``
  so downstream code can treat extracted precedents and identified
  risks symmetrically.
* No EN reference (we don't have one for new chapters). The scanner
  is operating on JP source alone.
* Cached at ``cache_root/risk-cache/{part_id}.json`` like extraction so
  re-running is free unless ``force=True``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from translator.inference.provider import complete
from translator.precedents.extract import default_v2_root
from translator.prep.corpus import load_part_jp


# Sonnet by default. Override with ``--model`` (e.g. deepseek-v4-pro).
DEFAULT_RISK_MODEL = "claude-sonnet-4-6"

# Output cap: a 5K-char chapter typically yields 20-50 risk entries
# at < 200 chars per record. 16K is generous headroom.
_RISK_MAX_TOKENS = 16_384

# Same categories as extraction so retrieval can correlate.
_VALID_CATEGORIES = {
    "facial_idiom", "set_phrase", "restructured",
    "condensed", "cultural", "other",
}

_RISK_LEVELS = {"high", "medium", "low"}


@dataclass
class RiskEntry:
    """One JP phrase flagged as likely-translationese-prone."""

    jp_span: str
    category: str  # one of _VALID_CATEGORIES
    risk_level: str  # high | medium | low
    literal_trap: str  # what a naive translator would write
    reason: str  # why the literal is wrong / why it's risky
    context_jp: str  # surrounding paragraph for retrieval

    def to_record(self) -> dict:
        return {
            "jp_span": self.jp_span,
            "category": self.category,
            "risk_level": self.risk_level,
            "literal_trap": self.literal_trap,
            "reason": self.reason,
            "context_jp": self.context_jp,
        }


@dataclass
class ChapterRisks:
    part_id: str
    risks: list[RiskEntry] = field(default_factory=list)


_SYSTEM_PROMPT = """\
You are a senior JP→EN literary translator with deep familiarity with \
Japanese light-novel idioms, set phrases, and restructuring conventions.

Your job is *risk assessment*: given the Japanese source of an \
untranslated chapter, identify the specific JP phrases that an \
inexperienced translator would render literally and produce \
translationese. You are NOT translating; you are flagging.

Output strict JSON, no markdown, no commentary, no code fences. \
String escaping must be valid JSON.
"""


_USER_PROMPT_TEMPLATE = """\
# Task

Scan the Japanese chapter below and identify phrases at risk of \
producing translationese.

## What to flag

A phrase is "at risk" if a literal word-for-word rendering would be \
stilted, unidiomatic, or technically correct but unnatural English. \
The categories (same vocabulary as our precedent index):

- ``facial_idiom``: 口角を上げる / 眉をひそめる / 目を細める / etc. \
where the literal physical description is wrong; English uses verbs \
like "smiled", "frowned", "squinted".
- ``set_phrase``: stock JP phrases (予想の範囲内, 言うまでもなく, \
無難に終わる, 当たり前 etc.) that have clean idiomatic English equivalents.
- ``restructured``: phrases where natural English requires \
restructuring the grammar (subject changes, voice flip, condensed clauses).
- ``condensed``: JP phrases that English natively expresses in fewer \
words / one verb (余裕だった → "called it easy"; よくわからない → \
"can't tell").
- ``cultural``: culture-specific terms (制服, 部活, 先輩, family-name \
+ さん addressing, etc.) where context determines whether to romanize, \
calque, or paraphrase.
- ``other``: any other phrase you'd flag from experience.

Skip phrases that translate cleanly word-for-word. Skip standard \
vocabulary, proper nouns, and generic dialogue. Be precise — the \
downstream consumer is the translator who will look up established \
precedents for each flagged span.

For each entry:
- ``jp_span``: the exact JP substring (verbatim from the source).
- ``category``: one of the categories above.
- ``risk_level``: ``"high"`` (this WILL produce translationese unless \
addressed), ``"medium"`` (likely but a careful translator could land \
it), ``"low"`` (worth flagging but minor).
- ``literal_trap``: the stiff word-for-word EN that an inexperienced \
translator would produce. Be concrete.
- ``reason``: one sentence explaining why the literal is wrong.
- ``context_jp``: the full JP paragraph this span appears in (verbatim, \
so retrieval can use the surrounding context).

Be exhaustive within the category boundaries. A 2000-3000 char chapter \
typically yields 25-60 flagged phrases. A risk-light chapter of pure \
dialogue might yield fewer.

## Output schema (strict JSON, no markdown, no commentary)

{{
  "risks": [
    {{
      "jp_span": "...",
      "category": "facial_idiom",
      "risk_level": "high",
      "literal_trap": "...",
      "reason": "...",
      "context_jp": "..."
    }}
  ]
}}

# Chapter

{jp_text}
"""


def _build_prompt(jp_text: str) -> str:
    return _USER_PROMPT_TEMPLATE.format(jp_text=jp_text)


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences if the model wrapped its JSON."""
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (``` or ```json) and trailing fence.
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_response(raw: str, *, part_id: str) -> ChapterRisks:
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:500].replace("\n", " ")
        raise ValueError(
            f"Failed to parse risk-scan JSON for {part_id}: {exc}\n"
            f"--- raw output (first 500 chars) ---\n{preview}"
        ) from exc

    risks: list[RiskEntry] = []
    for r in data.get("risks", []):
        try:
            cat = str(r.get("category", "other")).strip().lower()
            level = str(r.get("risk_level", "medium")).strip().lower()
            if cat not in _VALID_CATEGORIES:
                cat = "other"
            if level not in _RISK_LEVELS:
                level = "medium"
            risks.append(
                RiskEntry(
                    jp_span=str(r.get("jp_span", "")).strip(),
                    category=cat,
                    risk_level=level,
                    literal_trap=str(r.get("literal_trap", "")).strip(),
                    reason=str(r.get("reason", "")).strip(),
                    context_jp=str(r.get("context_jp", "")).strip(),
                )
            )
        except Exception:
            continue

    risks = [r for r in risks if r.jp_span]
    return ChapterRisks(part_id=part_id, risks=risks)


def _filter_verbatim(risks: ChapterRisks, jp_source: str) -> ChapterRisks:
    """Drop risks whose ``jp_span`` is not a verbatim substring of source.

    Mirrors the verbatim filter used in extraction. Keeps the index
    citable: every flagged span is something the translator can locate
    in the chapter.
    """
    src_norm = "".join(jp_source.split()).lower()
    kept: list[RiskEntry] = []
    for r in risks.risks:
        if "".join(r.jp_span.split()).lower() in src_norm:
            kept.append(r)
    return ChapterRisks(part_id=risks.part_id, risks=kept)


def _risk_cache_path(root: Path, part_id: str) -> Path:
    return root / "risk-cache" / f"{part_id}.json"


def scan_chapter(
    part_id: str,
    *,
    model: str = DEFAULT_RISK_MODEL,
    cache_root: Path | None = None,
    force: bool = False,
) -> ChapterRisks:
    """Run the at-risk scanner on a single chapter.

    Caches raw LLM output at ``cache_root/risk-cache/{part_id}.json``.
    """
    root = cache_root or default_v2_root()
    cp = _risk_cache_path(root, part_id)

    jp_text = load_part_jp(part_id)
    if not jp_text:
        return ChapterRisks(part_id=part_id)

    if not force and cp.exists():
        cached = json.loads(cp.read_text(encoding="utf-8"))
        candidate = _parse_response(cached["raw_response"], part_id=part_id)
        return _filter_verbatim(candidate, jp_text)

    prompt = _build_prompt(jp_text)
    last_exc: Exception | None = None
    response = ""
    candidate: ChapterRisks | None = None
    for attempt in range(3):
        temp = [0.2, 0.5, 0.7][attempt]
        response = complete(
            model=model,
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=_RISK_MAX_TOKENS,
            json_schema={"type": "object"},
        )
        try:
            candidate = _parse_response(response, part_id=part_id)
            break
        except ValueError as exc:
            last_exc = exc
            continue
    if candidate is None:
        assert last_exc is not None
        raise last_exc

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        json.dumps(
            {
                "part_id": part_id,
                "model": model,
                "scanned_at": datetime.now(UTC).isoformat(),
                "raw_response": response,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return _filter_verbatim(candidate, jp_text)


__all__ = [
    "ChapterRisks",
    "DEFAULT_RISK_MODEL",
    "RiskEntry",
    "scan_chapter",
]
